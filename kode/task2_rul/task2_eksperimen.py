"""Task 2 — runner eksperimen RUL: perbandingan model, ketidakpastian, aturan servis, final.

Grup (--grup):
  tune    : tuning resep DL di fold 0 (t2_tune_*), GRU awal sebelum tuning, LightGBM W80
  model   : GRU, TCN, Transformer, baseline LightGBM — validasi silang 5 fold per mesin (2d, 2e)
  window  : panjang window 30 vs 50 untuk model terbaik
  uncert  : fold 0, ensemble 5 seed x MC dropout 30 -> interval 90%, cakupan, kalibrasi
            conformal, aturan servis (2f)
  final   : 5 seed dilatih di semua mesin latih (+ MC dropout) -> prediksi_rul.csv
  tabel   : hanya ringkasan

Ketidakpastian (2f):
- Tiap anggota ensemble memberi mean & std MC dropout. Prediksi gabungan = campuran:
  mu = rata-rata mu_i, var = rata-rata(sd_i^2 + mu_i^2) - mu^2 (varians epistemik antar-seed +
  varians MC dropout). Interval awal = mu +- 1.645*sd.
- Interval seperti ini biasanya terlalu sempit, jadi dikalibrasi dengan split-conformal:
  skor r = |y - mu| / sd pada separuh mesin validasi; faktor q = kuantil 90% dari r; interval =
  mu +- q*sd; cakupan dicek di separuh lainnya (lalu ditukar). Faktor q dari seluruh fold 0
  dipakai untuk interval data uji.

    python kode/task2_eksperimen.py --grup model
"""
import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# folder task ini (modul task) + kode/ (paket common, metrik) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import ROOT, baca_json, buat_run, config_beda, folder_figur, simpan_metrics  # noqa: E402
from metrik import RUL_CAP, cakupan_interval, ringkasan_rul, rmse, skor_asimetris  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
Z90 = 1.6449


def jalankan(nama, argumen, data, ulang=False):
    if not ulang and os.path.exists(os.path.join(RUNS, nama, "metrics.json")):
        beda = config_beda(os.path.join(RUNS, nama), argumen)
        if not beda:
            print(f"[lewati] {nama}")
            return
        print(f"[ULANG] {nama} ada tetapi konfigurasinya berbeda: {beda}", flush=True)
    cmd = [sys.executable, os.path.join(HERE, "task2_train.py"), "--data", data, "--nama", nama] + argumen
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    # Di Windows proses CUDA kadang crash SAAT KELUAR (0xC0000409) setelah semua file tersimpan;
    # yang penting metrics.json sudah ditulis.
    if not os.path.exists(os.path.join(RUNS, nama, "metrics.json")):
        raise RuntimeError(f"{nama} gagal (exit {r.returncode})")


def tabel_model(nama_runs):
    rows = []
    for n in nama_runs:
        p = os.path.join(RUNS, n, "metrics.json")
        if not os.path.exists(p):
            continue
        m = baca_json(p)
        tp = m["titik_potong"]
        r = {"run": n, "rmse": tp["rmse"], "skor_total": tp["skor"], "skor_per_titik": tp["skor_per_mesin"],
             "rmse_fold_mean": m["rmse_potong_mean_fold"], "rmse_fold_std": m["rmse_potong_std_fold"],
             "rmse_semua_titik": m["semua_titik"]["rmse"], "waktu_s": m["waktu_detik"]}
        for rg in ["rul_0_50", "rul_50_100", "rul_100_150"]:
            r[f"rmse_{rg}"] = tp[rg]["rmse"]
            r[f"skor_{rg}"] = tp[rg]["skor"]
            r[f"n_{rg}"] = tp[rg]["n"]
        rows.append(r)
    return pd.DataFrame(rows)


def gabung_ensemble(runs, fold_only=True):
    dfs = [pd.read_csv(os.path.join(RUNS, n, "oof.csv.gz")) for n in runs]
    base = dfs[0][["id_unit", "siklus", "rul", "potong"]].copy()
    mus = np.stack([d["pred"].values for d in dfs])
    sds = np.stack([d["sd"].values for d in dfs])
    base["mu"] = mus.mean(0)
    base["sd_mc"] = np.sqrt((sds ** 2).mean(0))
    base["sd_ens"] = mus.std(0)
    base["sd"] = np.sqrt(np.maximum((sds ** 2 + mus ** 2).mean(0) - base["mu"] ** 2, 1e-6))
    base["mu_s0"], base["sd_s0"] = mus[0], sds[0]
    return base


def kuantil_conformal(r, alpha=0.1):
    n = len(r)
    return float(np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))


def analisis_uncert(model, data):
    run = buat_run(f"t2_uncert_{model}", {"model": model})
    runs = [f"t2_unc_{model}_s{s}" for s in range(5)]
    df = gabung_ensemble(runs)
    pc = df[df.potong].copy()
    y = pc["rul"].values
    hasil = {"n_titik_potong": int(len(pc)), "n_mesin": int(pc.id_unit.nunique())}
    # akurasi titik: seed tunggal vs ensemble
    hasil["rmse_seed0"] = rmse(y, np.clip(pc.mu_s0, 0, RUL_CAP))
    hasil["rmse_ensemble"] = rmse(y, np.clip(pc.mu, 0, RUL_CAP))
    hasil["skor_seed0"] = skor_asimetris(y, np.clip(pc.mu_s0, 0, RUL_CAP))
    hasil["skor_ensemble"] = skor_asimetris(y, np.clip(pc.mu, 0, RUL_CAP))
    # cakupan interval mentah (90% nominal)
    for nm, sd in [("mc_dropout_seed0", pc.sd_s0), ("ensemble_saja", pc.sd_ens), ("ensemble+mc", pc.sd)]:
        mu = pc.mu_s0 if nm.startswith("mc") else pc.mu
        lo, hi = np.clip(mu - Z90 * sd, 0, RUL_CAP), np.clip(mu + Z90 * sd, 0, RUL_CAP)
        hasil[f"cakupan_mentah_{nm}"] = cakupan_interval(y, lo, hi)
        hasil[f"lebar_mentah_{nm}"] = float((hi - lo).mean())
    # split conformal (2 arah)
    unit = pc.id_unit.unique()
    rng = np.random.RandomState(0)
    rng.shuffle(unit)
    A, B = set(unit[: len(unit) // 2]), set(unit[len(unit) // 2:])
    r = (np.abs(pc.rul - pc.mu) / pc.sd).values
    cak, leb = [], []
    for cal, tes in [(A, B), (B, A)]:
        mc, mt = pc.id_unit.isin(cal).values, pc.id_unit.isin(tes).values
        q = kuantil_conformal(r[mc])
        lo, hi = np.clip(pc.mu[mt] - q * pc.sd[mt], 0, RUL_CAP), np.clip(pc.mu[mt] + q * pc.sd[mt], 0, RUL_CAP)
        cak.append(cakupan_interval(y[mt], lo, hi)); leb.append(float((hi - lo).mean()))
    hasil["cakupan_conformal_split"] = float(np.mean(cak))
    hasil["lebar_conformal_split"] = float(np.mean(leb))
    q_all = kuantil_conformal(r)
    hasil["faktor_conformal_q"] = q_all
    pc["lo"], pc["hi"] = np.clip(pc.mu - q_all * pc.sd, 0, RUL_CAP), np.clip(pc.mu + q_all * pc.sd, 0, RUL_CAP)
    hasil["cakupan_per_rentang"] = {f"{a}-{b}": cakupan_interval(y[(y >= a) & (y <= b)], pc.lo[(y >= a) & (y <= b)],
                                                                 pc.hi[(y >= a) & (y <= b)]) for a, b in [(0, 50), (50, 100), (100, 150)]}
    hasil["korelasi_sd_vs_galat_abs"] = float(np.corrcoef(pc.sd, np.abs(pc.rul - pc.mu))[0, 1])

    # aturan servis: servis bila (a) prediksi titik <= T, atau (b) batas bawah 90% <= T
    aturan = []
    for T in [10, 20, 30, 40]:
        for nm, v in [("prediksi_titik", pc.mu), ("batas_bawah_90", pc.lo)]:
            servis = v.values <= T
            gagal = y <= 10                    # akan rusak <= 10 siklus lagi
            aman = y > 50
            aturan.append({"T": T, "aturan": nm,
                           "tertangkap_rusak<=10": float(servis[gagal].mean()) if gagal.any() else None,
                           "alarm_palsu_rul>50": float(servis[aman].mean()),
                           "frac_diservis": float(servis.mean()),
                           "rata2_sisa_umur_terbuang": float(y[servis].mean()) if servis.any() else None})
    hasil["aturan_servis"] = aturan

    # koreksi bias asimetris: geser prediksi c siklus, pilih c meminimalkan skor
    cs = np.arange(-20, 6)
    sk = [skor_asimetris(y, np.clip(pc.mu + c, 0, RUL_CAP)) for c in cs]
    rm = [rmse(y, np.clip(pc.mu + c, 0, RUL_CAP)) for c in cs]
    hasil["geser_bias"] = {"c_terbaik_skor": int(cs[int(np.argmin(sk))]), "skor_c0": sk[list(cs).index(0)],
                           "skor_terbaik": float(min(sk)), "rmse_c0": rm[list(cs).index(0)],
                           "rmse_di_c_terbaik": rm[int(np.argmin(sk))]}

    # figur: lintasan prediksi 4 mesin validasi + reliabilitas
    fd = folder_figur(2)
    df["lo"], df["hi"] = np.clip(df.mu - q_all * df.sd, 0, RUL_CAP), np.clip(df.mu + q_all * df.sd, 0, RUL_CAP)
    pilih = df.groupby("id_unit").siklus.max().sort_values().index[[10, 60, 120, -10]]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for axi, u in zip(ax, pilih):
        d = df[df.id_unit == u]
        axi.fill_between(d.siklus, d.lo, d.hi, alpha=.3, label="interval 90% (conformal)")
        axi.plot(d.siklus, d.mu, label="prediksi ensemble")
        axi.plot(d.siklus, d.rul, "k--", label="RUL aktual (cap 150)")
        axi.axhline(30, color="r", lw=.6, ls=":")
        axi.set_title(f"mesin {u}"); axi.set_xlabel("siklus")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "uncert_lintasan.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(pc.rul, pc.mu, yerr=q_all * pc.sd, fmt=".", ms=2, alpha=.3, elinewidth=.3)
    ax[0].plot([0, 150], [0, 150], "k--"); ax[0].set_xlabel("RUL aktual"); ax[0].set_ylabel("prediksi")
    ax[0].set_title("Titik potong validasi + interval 90%")
    lv = np.linspace(0.1, 0.99, 15)
    from scipy.stats import norm
    emp_raw = [np.mean(np.abs(pc.rul - pc.mu) <= norm.ppf(0.5 + l / 2) * pc.sd) for l in lv]
    emp_cal = [np.mean(np.abs(pc.rul - pc.mu) <= norm.ppf(0.5 + l / 2) * pc.sd * q_all / Z90) for l in lv]
    ax[1].plot(lv, emp_raw, "o-", label="mentah (ensemble+MC)")
    ax[1].plot(lv, emp_cal, "s-", label="setelah conformal")
    ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set_xlabel("tingkat nominal"); ax[1].set_ylabel("cakupan empiris")
    ax[1].legend(); ax[1].set_title("Kalibrasi interval")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "uncert_kalibrasi.png"), dpi=110); plt.close(fig)
    simpan_metrics(run, hasil)
    import json
    print(json.dumps(hasil, indent=1, default=str))
    return hasil


def buat_prediksi(model, n_seed, data, geser=0.0):
    """Ensemble seed full -> prediksi_rul.csv (rul = mean, clip 0-150) + interval di runs/t2_final.
    Prediksi dihitung ulang dari model tersimpan dengan seed MC dropout tetap (prediksi_uji_final), fungsi
    yang sama dengan reproduce.py, sehingga file yang dihasilkan identik."""
    import torch

    from task2_train import prediksi_uji_final

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = baca_json(os.path.join(RUNS, f"t2_full_{model}_s0", "config.json"))["argumen"]["W"]
    ids, mus, sds = prediksi_uji_final(model, W, [os.path.join(RUNS, f"t2_full_{model}_s{s}", "model.pt")
                                                  for s in range(n_seed)],
                                       os.path.join(RUNS, f"t2_full_{model}_s0", "prep.pkl"), data, dev)
    dfs = [pd.DataFrame({"id_unit": ids})]
    mu = mus.mean(0)
    sd = np.sqrt(np.maximum((sds ** 2 + mus ** 2).mean(0) - mu ** 2, 1e-6))
    pu = os.path.join(RUNS, f"t2_uncert_{model}", "metrics.json")
    q = baca_json(pu)["faktor_conformal_q"] if os.path.exists(pu) else Z90
    rul = np.clip(mu + geser, 0, RUL_CAP)
    out = pd.DataFrame({"id_unit": dfs[0].id_unit, "rul": np.round(rul, 2)})
    out.to_csv(os.path.join(ROOT, "prediksi_rul.csv"), index=False)
    run = buat_run("t2_final", {"model": model, "n_seed": n_seed, "geser": geser})
    detail = out.assign(batas_bawah_90=np.clip(mu - q * sd, 0, RUL_CAP).round(2),
                        batas_atas_90=np.clip(mu + q * sd, 0, RUL_CAP).round(2),
                        servis_sekarang=(np.clip(mu - q * sd, 0, RUL_CAP) <= 30))
    detail.to_csv(os.path.join(run, "prediksi_rul_dengan_interval.csv"), index=False)
    simpan_metrics(run, {"n_mesin": len(out), "rul_rata2": float(rul.mean()), "faktor_q": q,
                         "frac_servis_sekarang": float(detail.servis_sekarang.mean()), "geser": geser})
    print("prediksi_rul.csv:", out.rul.describe().round(2).to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--grup", required=True, choices=["tune", "model", "window", "uncert", "final", "tabel", "prediksi"])
    ap.add_argument("--model", default="gru", help="model terbaik untuk grup window/uncert/final")
    ap.add_argument("--W", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n_seed", type=int, default=5)
    ap.add_argument("--geser", type=float, default=0.0, help="koreksi bias prediksi final (siklus)")
    ap.add_argument("--ulang", action="store_true")
    a = ap.parse_args()
    # Resep DL hasil tuning di fold 0 (lihat t2_tune_*): W=50, 10 epoch, lr 1e-3, dropout 0.3,
    # noise input 0.3, weight decay 1e-3. Dipakai sama untuk ketiga arsitektur agar adil.
    ep = ["--epochs", str(a.epochs), "--lr", "1e-3", "--dropout", "0.3", "--noise", "0.3", "--wd", "1e-3"]
    if a.grup == "tune":
        # Tuning resep DL di fold 0 (dasar pemilihan resep `ep` di atas) + run awal sebelum tuning.
        # Argumen = persis seperti config.json run yang dilaporkan, agar bisa direproduksi satu perintah.
        A = ["--epochs", "15", "--lr", "2e-3", "--dropout", "0.3", "--wd", "1e-3", "--noise", "0.3"]   # resep A
        B = ["--epochs", "10", "--lr", "2e-3", "--dropout", "0.2", "--wd", "1e-4", "--noise", "0"]     # resep B
        f0 = ["--mode", "cv", "--fold", "0"]
        tune = {
            "t2_tune_gru_A": ["--model", "gru", "--W", "50"] + A,
            "t2_tune_gru_B": ["--model", "gru", "--W", "30"] + B,
            "t2_tune_gru_C": ["--model", "gru", "--W", "50", "--epochs", "25", "--lr", "2e-3", "--dropout", "0.3",
                              "--wd", "1e-3", "--noise", "0.5"],
            "t2_tune_gru_D": ["--model", "gru", "--W", "80"] + A,
            "t2_tune_gru_E": ["--model", "gru", "--W", "50"] + ep,
            "t2_tune_tcn_A": ["--model", "tcn", "--W", "50"] + A,
            "t2_tune_tcn_B": ["--model", "tcn", "--W", "30"] + B,
            "t2_tune_tf_A": ["--model", "transformer", "--W", "50"] + A,
            "t2_tune_tf_B": ["--model", "transformer", "--W", "30"] + B,
        }
        for nama, arg in tune.items():
            jalankan(nama, arg + f0, a.data, a.ulang)
        jalankan("t2_cv_gru_awal", ["--model", "gru", "--mode", "cv", "--W", "30", "--epochs", "25", "--lr", "2e-3"],
                 a.data, a.ulang)
        jalankan("t2_cv_lgbm_W80", ["--model", "lgbm", "--mode", "cv", "--W", "80"], a.data, a.ulang)
    elif a.grup == "model":
        jalankan("t2_cv_lgbm", ["--model", "lgbm", "--mode", "cv", "--W", "30"], a.data, a.ulang)
        jalankan("t2_cv_lgbm_W50", ["--model", "lgbm", "--mode", "cv", "--W", "50"], a.data, a.ulang)
        for m in ["gru", "tcn", "transformer"]:
            jalankan(f"t2_cv_{m}", ["--model", m, "--mode", "cv", "--W", str(a.W)] + ep, a.data, a.ulang)
    elif a.grup == "window":
        jalankan(f"t2_cv_{a.model}_W30", ["--model", a.model, "--mode", "cv", "--W", "30"] + ep, a.data, a.ulang)
    elif a.grup == "uncert":
        for s in range(5):
            jalankan(f"t2_unc_{a.model}_s{s}", ["--model", a.model, "--mode", "cv", "--fold", "0", "--seed", str(s),
                                                "--mc", "30", "--W", str(a.W)] + ep, a.data, a.ulang)
        analisis_uncert(a.model, a.data)
    elif a.grup == "final":
        for s in range(a.n_seed):
            jalankan(f"t2_full_{a.model}_s{s}", ["--model", a.model, "--mode", "full", "--seed", str(s), "--mc", "30",
                                                 "--W", str(a.W)] + ep, a.data, a.ulang)
        buat_prediksi(a.model, a.n_seed, a.data, a.geser)
    elif a.grup == "prediksi":
        buat_prediksi(a.model, a.n_seed, a.data, a.geser)
    names = ["t2_cv_lgbm", "t2_cv_lgbm_W50", "t2_cv_gru_awal"] + [f"t2_cv_{m}" for m in ["gru", "tcn", "transformer"]] +             [f"t2_cv_{m}_W30" for m in ["gru", "tcn", "transformer"]]
    df = tabel_model(names)
    if len(df):
        os.makedirs(os.path.join(RUNS, "t2_tabel"), exist_ok=True)
        df.to_csv(os.path.join(RUNS, "t2_tabel", "model.csv"), index=False)
        print(df[["run", "rmse", "skor_total", "skor_per_titik", "rmse_fold_mean", "rmse_fold_std",
                  "rmse_rul_0_50", "rmse_rul_50_100", "rmse_rul_100_150", "waktu_s"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
