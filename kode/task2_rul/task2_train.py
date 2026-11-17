"""Task 2d-e — training & validasi model RUL (GRU / TCN / Transformer / baseline LightGBM).

Validasi (--mode cv): GroupKFold 5 fold PER MESIN (bukan per baris): semua siklus satu mesin
berada di fold yang sama, sehingga model tidak pernah melihat mesin validasi saat training.
Preprocessing (mean/std per regime) di-fit ulang di setiap fold latih.

Evaluasi utama meniru data uji: data uji berhenti di titik acak sebelum rusak, jadi untuk tiap
mesin validasi diambil 5 "titik potong" acak (siklus >= 50) dan RUL diprediksi dari window
yang BERAKHIR di titik itu (tidak melihat siklus sesudahnya). Dilaporkan RMSE dan skor
asimetris total + per rentang RUL aktual (0-50, 50-100, 100-150). RMSE di semua titik juga
disimpan sebagai pembanding.

Mode final (--mode full): latih di seluruh 900 mesin, prediksi RUL di siklus terakhir tiap
mesin uji (opsi --mc N untuk MC dropout).

  python kode/task2_train.py --model gru --mode cv --nama t2_cv_gru
  python kode/task2_train.py --model lgbm --mode cv --nama t2_cv_lgbm
  python kode/task2_train.py --model gru --mode full --seed 0 --mc 30 --nama t2_full_gru_s0
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# folder task ini (modul task) + kode/ (paket common, metrik) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import buat_run, set_seed, simpan_json, simpan_metrics  # noqa: E402
from metrik import RUL_CAP, ringkasan_rul, rmse  # noqa: E402
from task2_prep import Jendela, Prep, baris_terakhir, label_rul, muat, titik_potong  # noqa: E402


# ------------------------------------------------------------------ baseline non-DL
def fitur_tabular(J, idx, n_sensor):
    """Ringkasan window untuk LightGBM: nilai terakhir, rata-rata, std, kemiringan tiap sensor
    (hanya siklus asli), + siklus & regime saat ini."""
    X = J.numpy_batch(idx)                      # (B, W, F+1)
    m = X[:, :, -1:]                            # mask
    s = X[:, :, :n_sensor]
    n = m.sum(1).clip(min=1)
    mean = (s * m).sum(1) / n
    std = np.sqrt((((s - mean[:, None]) ** 2) * m).sum(1) / n)
    t = np.arange(X.shape[1], dtype=np.float32)[None, :, None]
    tm = (t * m).sum(1) / n
    cov = (((t - tm[:, None]) * (s - mean[:, None])) * m).sum(1)
    var = (((t - tm[:, None]) ** 2) * m).sum(1).clip(min=1e-6)
    slope = cov / var
    last = X[:, -1, :-1]                        # sensor terakhir + siklus + regime
    return np.concatenate([last, mean, std, slope, n], 1)


def latih_lgbm(Jtr, idx_tr, y_tr, n_sensor, seed):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(fitur_tabular(Jtr, idx_tr, n_sensor), y_tr)
    return model


# ------------------------------------------------------------------ deep learning
def latih_dl(args, Jtr, y_tr, seed, dev, log=print):
    import torch

    from task2_model import buat_model

    set_seed(seed)
    model = buat_model(args.model, Jtr.X.shape[1], args.W, getattr(args, "dropout", 0.2)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, "wd", 1e-4))
    n_sensor = Jtr.X.shape[1] - 1 - 6 - 1           # kanal sensor (tanpa siklus, regime, mask)
    noise = getattr(args, "noise", 0.0)
    n = len(y_tr)
    langkah = args.epochs * int(np.ceil(n / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=langkah, pct_start=0.1)
    yt = torch.from_numpy(y_tr / RUL_CAP).float().to(dev)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            x = Jtr.batch(b)
            if noise > 0:  # augmentasi: noise Gaussian pada kanal sensor yang asli (bukan padding)
                x = x.clone()
                x[:, :, :n_sensor] += noise * torch.randn_like(x[:, :, :n_sensor]) * x[:, :, -1:]
            loss = torch.nn.functional.mse_loss(model(x), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(b)
        hist.append({"epoch": ep, "train_rmse": float(np.sqrt(tot / n)) * RUL_CAP})
        if ep % 5 == 0 or ep == args.epochs:
            log(f"   ep {ep:3d} train RMSE {hist[-1]['train_rmse']:.2f}")
    return model, hist


def prediksi_dl(model, J, idx, dev, mc=0, bs=4096):
    """Kembalikan (mean, std). mc=0: deterministik (std=0). mc>0: MC dropout mc kali."""
    import torch

    from task2_model import aktifkan_mc_dropout

    idx_t = torch.from_numpy(np.asarray(idx)).to(dev)
    with torch.no_grad():
        if mc == 0:
            model.eval()
            out = torch.cat([model(J.batch(idx_t[i:i + bs])) for i in range(0, len(idx_t), bs)])
            return out.cpu().numpy() * RUL_CAP, np.zeros(len(idx_t), np.float32)
        aktifkan_mc_dropout(model)
        sampel = []
        for _ in range(mc):
            sampel.append(torch.cat([model(J.batch(idx_t[i:i + bs])) for i in range(0, len(idx_t), bs)]))
        s = torch.stack(sampel).cpu().numpy() * RUL_CAP
        return s.mean(0), s.std(0)


def prediksi_uji_final(model_name, W, path_model, path_prep, data, dev, mc=30):
    """Prediksi data uji (baris terakhir tiap mesin) dari model final tersimpan, MC dropout dengan seed TETAP
    per model (1000 + indeks), sehingga `prediksi_rul.csv` dari runner eksperimen dan dari reproduce.py
    identik. Kembalikan (id_unit, mus (n_model, n_mesin), sds)."""
    import torch

    from task2_model import buat_model

    dte = muat(data, "test")
    with open(path_prep, "rb") as f:
        prep = pickle.load(f)
    J = Jendela(prep.transform(dte), dte["id_unit"].values, W).ke_torch(dev)
    akhir = baris_terakhir(dte)
    mus, sds = [], []
    for i, p in enumerate(path_model):
        ck = torch.load(p, map_location=dev, weights_only=False)
        # dropout HARUS sama dengan saat training (MC dropout memakai peluang ini)
        model = buat_model(model_name, J.X.shape[1], W, ck["args"].get("dropout", 0.2)).to(dev)
        model.load_state_dict(ck["state_dict"])
        set_seed(1000 + i)
        mu, sd = prediksi_dl(model, J, akhir, dev, mc=mc)
        mus.append(mu); sds.append(sd)
    return dte["id_unit"].values[akhir], np.array(mus), np.array(sds)


def satu_fold(args, df_tr, df_va, seed, dev, log=print):
    prep = Prep().fit(df_tr)
    ftr, info_tr = prep.transform(df_tr, kembalikan_info=True)
    fva = prep.transform(df_va)
    Jtr = Jendela(ftr, df_tr["id_unit"].values, args.W)
    Jva = Jendela(fva, df_va["id_unit"].values, args.W)
    y_tr = label_rul(df_tr)
    idx_tr = np.arange(len(df_tr))
    if args.model == "lgbm":
        m = latih_lgbm(Jtr, idx_tr, y_tr, len(prep.sensor), seed)
        pred = m.predict(fitur_tabular(Jva, np.arange(len(df_va)), len(prep.sensor)))
        return pred, np.zeros_like(pred), prep, m, [], info_tr
    Jtr.ke_torch(dev); Jva.ke_torch(dev)
    model, hist = latih_dl(args, Jtr, y_tr, seed, dev, log)
    pred, sd = prediksi_dl(model, Jva, np.arange(len(df_va)), dev, args.mc)
    return pred, sd, prep, model, hist, info_tr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--nama", default=None)
    ap.add_argument("--model", choices=["gru", "tcn", "transformer", "lgbm"], default="gru")
    ap.add_argument("--mode", choices=["cv", "full"], default="cv")
    ap.add_argument("--W", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.0, help="std noise augmentasi pada sensor (satuan z)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=-1, help="-1 = semua fold; k = hanya fold k")
    ap.add_argument("--mc", type=int, default=0, help="jumlah sampel MC dropout saat prediksi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    a.nama = a.nama or f"t2_{a.mode}_{a.model}_s{a.seed}" + ("_smoke" if a.smoke else "")
    set_seed(a.seed)
    run = buat_run(a.nama, a)
    t0 = time.time()
    dev = None
    if a.model != "lgbm":
        from common import get_device

        dev = get_device()

    df = muat(a.data, "train")
    if a.smoke:
        df = df[df["id_unit"].isin(df["id_unit"].unique()[:60])].reset_index(drop=True)
        a.epochs = 2

    if a.mode == "full":
        dte = muat(a.data, "test")
        pred, sd, prep, model, hist, info = satu_fold(a, df, dte, a.seed, dev)
        akhir = baris_terakhir(dte)
        out = pd.DataFrame({"id_unit": dte["id_unit"].values[akhir], "siklus": dte["siklus"].values[akhir],
                            "rul_mean": pred[akhir], "rul_sd": sd[akhir]})
        out.to_csv(os.path.join(run, "test_pred.csv"), index=False)
        with open(os.path.join(run, "prep.pkl"), "wb") as f:
            pickle.dump(prep, f)
        if a.model == "lgbm":
            with open(os.path.join(run, "model.pkl"), "wb") as f:
                pickle.dump(model, f)
        else:
            import torch

            torch.save({"state_dict": model.state_dict(), "args": vars(a)}, os.path.join(run, "model.pt"))
        simpan_json(hist, os.path.join(run, "history.json"))
        simpan_metrics(run, {"mode": "full", "n_mesin_train": int(df["id_unit"].nunique()), **info,
                             "pred_test_mean": float(out["rul_mean"].mean()),
                             "waktu_detik": round(time.time() - t0, 1)})
        print(f"[{a.nama}] selesai {time.time() - t0:.0f}s, prediksi test rata-rata {out['rul_mean'].mean():.1f}")
        return

    # ---- validasi silang per mesin
    from sklearn.model_selection import GroupKFold

    y = label_rul(df)
    potong = titik_potong(df, seed=0)
    is_potong = np.zeros(len(df), bool)
    is_potong[potong] = True
    oof = np.full(len(df), np.nan, np.float32)
    oof_sd = np.full(len(df), np.nan, np.float32)
    per_fold, hist_all = [], {}
    folds = list(GroupKFold(a.folds).split(df, groups=df["id_unit"]))
    for k, (itr, iva) in enumerate(folds):
        if a.fold >= 0 and k != a.fold:
            continue
        t1 = time.time()
        print(f"[{a.nama}] fold {k}: {df['id_unit'].iloc[itr].nunique()} mesin latih, {df['id_unit'].iloc[iva].nunique()} validasi")
        df_tr, df_va = df.iloc[itr].reset_index(drop=True), df.iloc[iva].reset_index(drop=True)
        pred, sd, prep, model, hist, _ = satu_fold(a, df_tr, df_va, a.seed + 100 * k, dev)
        oof[iva], oof_sd[iva] = pred, sd
        with open(os.path.join(run, f"prep_fold{k}.pkl"), "wb") as f:
            pickle.dump(prep, f)
        if a.model == "lgbm":
            with open(os.path.join(run, f"model_fold{k}.pkl"), "wb") as f:
                pickle.dump(model, f)
        else:
            import torch

            torch.save({"state_dict": model.state_dict(), "args": vars(a)}, os.path.join(run, f"model_fold{k}.pt"))
        hist_all[k] = hist
        sel = is_potong[iva]
        r = ringkasan_rul(y[iva][sel], np.clip(pred[sel], 0, RUL_CAP))
        per_fold.append({"fold": k, "rmse_potong": r["rmse"], "skor_per_mesin_potong": r["skor_per_mesin"],
                         "rmse_semua": rmse(y[iva], np.clip(pred, 0, RUL_CAP)), "detik": round(time.time() - t1, 1)})
        print(f"   fold {k}: RMSE potong {r['rmse']:.2f} | skor/titik {r['skor_per_mesin']:.2f} | {time.time() - t1:.0f}s")

    ada = ~np.isnan(oof)
    sel = ada & is_potong
    pc = np.clip(oof, 0, RUL_CAP)
    m = {"titik_potong": ringkasan_rul(y[sel], pc[sel]),
         "semua_titik": {"rmse": rmse(y[ada], pc[ada]), "n": int(ada.sum())},
         "per_fold": per_fold,
         "rmse_potong_mean_fold": float(np.mean([f["rmse_potong"] for f in per_fold])),
         "rmse_potong_std_fold": float(np.std([f["rmse_potong"] for f in per_fold], ddof=1)) if len(per_fold) > 1 else None,
         "waktu_detik": round(time.time() - t0, 1)}
    pd.DataFrame({"id_unit": df["id_unit"], "siklus": df["siklus"], "rul": y, "pred": oof, "sd": oof_sd,
                  "potong": is_potong})[ada].to_csv(os.path.join(run, "oof.csv.gz"), index=False)
    simpan_json(hist_all, os.path.join(run, "history.json"))
    simpan_metrics(run, m)
    tp = m["titik_potong"]
    print(f"[{a.nama}] RMSE potong {tp['rmse']:.2f} | skor {tp['skor']:.1f} (per titik {tp['skor_per_mesin']:.2f}) | "
          f"RMSE semua {m['semua_titik']['rmse']:.2f} | {m['waktu_detik']}s")


if __name__ == "__main__":
    main()
