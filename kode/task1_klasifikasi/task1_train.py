"""Task 1 — training & evaluasi klasifikasi multimodal (satu skrip untuk semua eksperimen).

Contoh:
  python kode/task1_klasifikasi/task1_train.py --modal image --nama t1_image                  # image-only
  python kode/task1_klasifikasi/task1_train.py --modal fusion --fusion xattn --nama t1_xattn  # fusion
  python kode/task1_klasifikasi/task1_train.py --overfit_batch --nama t1_overfit              # sanity check
  python kode/task1_klasifikasi/task1_train.py --smoke                                        # uji cepat

Pilihan yang dipakai di semua run (dan alasannya):
- Split validasi 20% stratified per (kelas, lini) dengan seed split TETAP (0), sehingga semua
  eksperimen dibandingkan di himpunan validasi yang sama; --seed hanya mengubah inisialisasi
  dan urutan batch.
- Loss cross-entropy berbobot kelas (1/sqrt(frekuensi)), karena kelas normal ~50%.
- Augmentasi geometris selalu aktif (flip, rotasi 90 derajat): cacat tidak punya orientasi tetap.
- Model yang dievaluasi = model di epoch TERAKHIR (jadwal OneCycle, turun ala cosine), bukan epoch terbaik di
  validasi, agar angka validasi tidak bias oleh pemilihan epoch.

Metrik yang disimpan (runs/<nama>/metrics.json):
- macro_f1, F1 per kelas, macro-F1 per lini
- macro_f1_bobot_test: sampel validasi diberi bobot lini (proporsi test / proporsi latih),
  sebagai estimasi kinerja pada komposisi lini data uji (60% lini B)
- macro_f1_tanpa_sensor: semua sensor validasi sengaja dihapus (uji missing modality)
- macro_f1_sensor_hilang_asli: hanya unit validasi yang sensornya memang tidak terekam
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# folder task ini (modul task) + kode/ (paket common) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import ROOT, buat_run, get_device, set_seed, simpan_json, simpan_metrics  # noqa: E402
from metrik import KELAS, f1_per_kelas, ringkasan_klasifikasi  # noqa: E402
from task1_data import (LINI, ambil_batch, bobot_lini_test, hitung_statistik, muat,  # noqa: E402
                        split_validasi, tensor_gpu)
from task1_model import Task1Net, jumlah_parameter  # noqa: E402


# Augmentasi tekstur (domain shift lini B). Analisis Fourier gambar `normal` DATA LATIH
# (kode/task1_klasifikasi/task1_tekstur.py -> runs/t1_tekstur/metrics.json): gambar lini B punya GARIS HALUS BERARAH
# dengan periode ±4 px (cincin frekuensi 13-18 siklus/64 px; porsi energi cincin ±9,8% vs ±2,2% di
# lini A). Di dalam satu gambar energinya terkumpul di sekitar satu arah (±75% dalam ±20° dari puncak),
# tetapi arah itu ACAK antar gambar (histogram sudut puncak merata 0-180°). Selisih energi cincin
# B - A ±24 → tambahan std ±4,8 piksel (skala 0-255). Garis halus ini mirip goresan/retakan tipis,
# sehingga model dilatih dengan garis tiruan yang sama (arah acak per gambar).
TEKSTUR_FREK = (13.0, 18.0)
TEKSTUR_STD_PX = 4.8
TEKSTUR_SUDUT_SD = 22.0       # derajat, lebar irisan arah (Gaussian) di sekitar arah acak
_grid = {}


def butiran(n, dev):
    """Garis halus tiruan lini B: noise yang disaring ke cincin 13-18 siklus/64 px DAN ke irisan sudut
    (Gaussian sd 22°) di sekitar arah acak per gambar. Std 1 per gambar, bentuk (n,1,64,64)."""
    if dev not in _grid:
        f = torch.fft.fftfreq(64, d=1 / 64, device=dev)
        fy, fx = f[:, None].expand(64, 64), f[None, :].expand(64, 64)
        r = torch.sqrt(fx ** 2 + fy ** 2)
        _grid[dev] = (((r >= TEKSTUR_FREK[0]) & (r <= TEKSTUR_FREK[1])).float(),
                      torch.rad2deg(torch.atan2(fy, fx)) % 180)
    cincin, sudut = _grid[dev]
    th = torch.rand(n, 1, 1, 1, device=dev) * 180
    d = (sudut[None, None] - th + 90) % 180 - 90            # selisih sudut melingkar (periode 180°)
    filt = cincin * torch.exp(-0.5 * (d / TEKSTUR_SUDUT_SD) ** 2)
    z = torch.fft.ifft2(torch.fft.fft2(torch.randn(n, 1, 64, 64, device=dev)) * filt).real
    return z / z.std((2, 3), keepdim=True).clamp(min=1e-6)


def augmentasi(b, aug_domain, mod_drop, g, aug_tekstur=0.0):
    """Augmentasi di GPU. g = torch.Generator untuk reprodusibilitas."""
    img, sen, mask = b["img"], b["sen"], b["mask"].clone()
    n = img.shape[0]
    dev = img.device
    if aug_tekstur > 0:
        # garis halus tiruan lini B pada sebagian sampel (arahnya sudah acak, jadi aman sebelum flip/rotasi)
        pakai = (torch.rand(n, 1, 1, 1, device=dev) < aug_tekstur).float()
        kuat = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.5, 1.5) * TEKSTUR_STD_PX / 255.0
        img = img + pakai * kuat / b["img_sd"][:, None, None, None] * butiran(n, dev)
    # geometris: flip horizontal/vertikal + rotasi 90 derajat acak (per batch, murah)
    if torch.rand(1, generator=g).item() < 0.5:
        img = img.flip(3)
    if torch.rand(1, generator=g).item() < 0.5:
        img = img.flip(2)
    k = int(torch.randint(0, 4, (1,), generator=g).item())
    img = torch.rot90(img, k, (2, 3))
    if aug_domain:
        # fotometrik: kontras & kecerahan acak per sampel -> meniru beda lini A/B
        c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
        br = torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.5, 0.5)
        img = img * c + br
        # sensor: noise kecil + offset suhu acak (lini B ~2 C lebih panas)
        sen = sen + 0.05 * torch.randn_like(sen)
        sen[:, 0] = sen[:, 0] + torch.empty(n, 1, device=dev).uniform_(-1.0, 1.0)
    if mod_drop > 0:
        # modality dropout: seluruh sensor hilang (p) atau satu kanal hilang (p/5)
        semua = torch.rand(n, 1, device=dev) < mod_drop
        kanal = torch.rand(n, 3, device=dev) < mod_drop / 5
        mask = mask * (~semua).float() * (~kanal).float()
    return img, sen, mask


@torch.no_grad()
def prediksi(model, t, idx, bs=1024, hapus_sensor=False):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = ambil_batch(t, idx[i:i + bs])
        mask = torch.zeros_like(b["mask"]) if hapus_sensor else b["mask"]
        with torch.autocast("cuda", dtype=torch.float16, enabled=b["img"].is_cuda):
            out.append(F.softmax(model(b["img"], b["sen"], mask).float(), 1).cpu())
    return torch.cat(out).numpy()


def bobot_kelas(y, n_kelas=7):
    f = np.bincount(y, minlength=n_kelas).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(f, 1))
    return (w / w.mean()).astype(np.float32)


def kurva(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    if "val_loss" in hist[0]:
        ax[0].plot(ep, [h["val_loss"] for h in hist], label="validasi")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="akurasi train")
    if "val_f1" in hist[0]:
        ax[1].plot(ep, [h["val_f1"] for h in hist], label="macro-F1 validasi")
    ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def overfit_satu_batch(args, d, dev, run_dir):
    """Sanity check: model harus bisa menghafal 64 sampel (loss -> ~0, akurasi -> 1)."""
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(d["meta"]), 64, replace=False)
    st = hitung_statistik(d, idx)
    t = tensor_gpu(d, st, args.norm, dev)
    model = Task1Net(args.modal, args.fusion).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    b = ambil_batch(t, torch.from_numpy(idx).to(dev))
    hist = []
    model.train()
    for step in range(1, args.overfit_steps + 1):
        logit = model(b["img"], b["sen"], b["mask"])
        loss = F.cross_entropy(logit, b["y"])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0 or step == 1:
            # akurasi dari forward pass training yang sama (mode train): yang diuji adalah
            # kemampuan menghafal batch, bukan statistik BatchNorm mode eval
            acc = (logit.argmax(1) == b["y"]).float().mean().item()
            hist.append({"epoch": step, "train_loss": loss.item(), "train_acc": acc})
            print(f"step {step:4d} loss {loss.item():.4f} acc {acc:.3f}")
    kurva(hist, os.path.join(run_dir, "kurva_overfit.png"))
    simpan_json(hist, os.path.join(run_dir, "history.json"))
    simpan_metrics(run_dir, {"overfit_loss_akhir": hist[-1]["train_loss"], "overfit_acc_akhir": hist[-1]["train_acc"],
                             "n_sampel": 64, "langkah": args.overfit_steps})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--nama", default=None)
    ap.add_argument("--modal", choices=["image", "sensor", "fusion"], default="fusion")
    ap.add_argument("--fusion", choices=["concat", "late", "xattn"], default="concat")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--norm", choices=["global", "perline"], default="global")
    ap.add_argument("--aug", action="store_true", help="augmentasi fotometrik + offset sensor (domain)")
    ap.add_argument("--line_reweight", action="store_true", help="bobot loss per lini = p_test/p_train")
    ap.add_argument("--aug_tekstur", type=float, default=0.0,
                    help="peluang menambah garis halus tiruan lini B (periode ±4 px, arah acak) pada gambar latih (domain)")
    ap.add_argument("--mod_drop", type=float, default=0.0, help="peluang sensor dihapus saat training")
    ap.add_argument("--drop_ids", default=None, help="CSV id_unit yang dibuang dari training (label noise)")
    ap.add_argument("--kfold", type=int, default=0, help=">0: pakai fold ke --fold dari StratifiedKFold")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_test", action="store_true", help="simpan probabilitas data uji")
    ap.add_argument("--overfit_batch", action="store_true")
    ap.add_argument("--overfit_steps", type=int, default=300)
    ap.add_argument("--smoke", action="store_true", help="subset kecil, 2 epoch (uji kode)")
    ap.add_argument("--runs_dir", default=None)
    args = ap.parse_args(argv)
    if args.nama is None:
        args.nama = f"t1_{args.modal}_{args.fusion}_s{args.seed}" + ("_smoke" if args.smoke else "")
    set_seed(args.seed)
    dev = get_device()
    run_dir = buat_run(args.nama, args, args.runs_dir)
    t0 = time.time()

    d = muat(args.data, "train")
    if args.overfit_batch:
        return overfit_satu_batch(args, d, dev, run_dir)
    import pandas as pd

    dte = muat(args.data, "test") if args.save_test else None
    meta_te = dte["meta"] if dte is not None else pd.read_csv(os.path.join(args.data, "klasifikasi_test_meta.csv"))
    w_lini = bobot_lini_test(d["meta"], meta_te)

    if args.kfold > 0:
        from sklearn.model_selection import StratifiedKFold

        strat = d["meta"]["kelas"] + "_" + d["meta"]["lini"]
        folds = list(StratifiedKFold(args.kfold, shuffle=True, random_state=0).split(np.zeros(len(strat)), strat))
        idx_tr, idx_va = folds[args.fold]
    else:
        idx_tr, idx_va = split_validasi(d["meta"], 0.2, seed=0)
    if args.drop_ids:
        # path relatif diselesaikan terhadap root repo (mis. "runs/t1_noise/mungkin_salah_label.csv"),
        # supaya sama saja dijalankan dari folder mana pun
        p = args.drop_ids if os.path.isabs(args.drop_ids) else os.path.join(ROOT, args.drop_ids)
        buang = set(pd.read_csv(p)["id_unit"])
        sebelum = len(idx_tr)
        idx_tr = idx_tr[~d["meta"]["id_unit"].iloc[idx_tr].isin(buang).values]
        print(f"buang {sebelum - len(idx_tr)} sampel latih (kemungkinan salah label)")
    if args.smoke:
        rng = np.random.RandomState(0)
        idx_tr = rng.choice(idx_tr, min(2000, len(idx_tr)), replace=False)
        idx_va = rng.choice(idx_va, min(600, len(idx_va)), replace=False)
        args.epochs = 2

    st = hitung_statistik(d, idx_tr)
    t = tensor_gpu(d, st, args.norm, dev)
    y = d["y"]
    wk = torch.tensor(bobot_kelas(y[idx_tr]), device=dev)
    wl = torch.tensor(w_lini if args.line_reweight else np.ones(2, np.float32), device=dev)

    model = Task1Net(args.modal, args.fusion).to(dev)
    print(f"[{args.nama}] params={jumlah_parameter(model):,} train={len(idx_tr)} val={len(idx_va)} device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(idx_tr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    g = torch.Generator().manual_seed(args.seed)
    idx_tr_t = torch.from_numpy(idx_tr).to(dev)
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = idx_tr_t[torch.randperm(len(idx_tr_t), generator=g).to(dev)]
        tl, tc, tn = 0.0, 0, 0
        for i in range(0, len(perm), args.bs):
            b = ambil_batch(t, perm[i:i + args.bs])
            img, sen, mask = augmentasi(b, args.aug, args.mod_drop, g, args.aug_tekstur)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(img, sen, mask)
            loss_s = F.cross_entropy(logit.float(), b["y"], weight=wk, reduction="none", label_smoothing=args.ls)
            loss = (loss_s * wl[b["lini"]]).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(b["y"]); tc += (logit.argmax(1) == b["y"]).sum().item(); tn += len(b["y"])
        p_va = prediksi(model, t, torch.from_numpy(idx_va).to(dev))
        vl = float(F.nll_loss(torch.log(torch.from_numpy(p_va) + 1e-8), torch.from_numpy(y[idx_va])).item())
        vf1 = float(np.nanmean(f1_per_kelas(y[idx_va], p_va.argmax(1))))
        hist.append({"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vl, "val_f1": vf1})
        print(f"ep {ep:3d} loss {tl / tn:.4f} acc {tc / tn:.3f} | val loss {vl:.4f} macroF1 {vf1:.4f} | {time.time() - t0:.0f}s")

    # ---- evaluasi akhir (epoch terakhir)
    va_t = torch.from_numpy(idx_va).to(dev)
    p_va = prediksi(model, t, va_t)
    yv, pv = y[idx_va], p_va.argmax(1)
    lini_va = d["lini"][idx_va]
    m = ringkasan_klasifikasi(yv, pv, grup=np.array(LINI)[lini_va])
    m["macro_f1_bobot_test"] = float(np.nanmean(f1_per_kelas(yv, pv, bobot=w_lini[lini_va])))
    # F1 per kelas per lini: menunjukkan kelas mana yang paling terdampak domain shift
    m["f1_per_kelas_per_lini"] = {l: dict(zip(KELAS, map(float, f1_per_kelas(yv[lini_va == i], pv[lini_va == i]))))
                                  for i, l in enumerate(LINI)}
    p_ns = prediksi(model, t, va_t, hapus_sensor=True)
    m["macro_f1_tanpa_sensor"] = float(np.nanmean(f1_per_kelas(yv, p_ns.argmax(1))))
    m["f1_per_kelas_tanpa_sensor"] = dict(zip(KELAS, map(float, f1_per_kelas(yv, p_ns.argmax(1)))))
    hilang = d["sensor"][idx_va].reshape(len(idx_va), -1)
    hilang = np.isnan(hilang).all(1)
    m["n_val_sensor_hilang_asli"] = int(hilang.sum())
    m["macro_f1_sensor_hilang_asli"] = float(np.nanmean(f1_per_kelas(yv[hilang], pv[hilang]))) if hilang.any() else None
    m["epoch_terbaik_val_f1"] = int(np.argmax([h["val_f1"] for h in hist]) + 1)
    m["val_f1_terbaik"] = float(max(h["val_f1"] for h in hist))
    m["waktu_detik"] = round(time.time() - t0, 1)
    m["n_param"] = jumlah_parameter(model)
    from sklearn.metrics import confusion_matrix

    np.save(os.path.join(run_dir, "confusion.npy"), confusion_matrix(yv, pv, labels=range(7)))
    np.save(os.path.join(run_dir, "val_idx.npy"), idx_va)
    np.save(os.path.join(run_dir, "val_probs.npy"), p_va.astype(np.float32))
    simpan_json(hist, os.path.join(run_dir, "history.json"))
    kurva(hist, os.path.join(run_dir, "kurva.png"))
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "stats": st}, os.path.join(run_dir, "model.pt"))
    if dte is not None:
        tt = tensor_gpu(dte, st, args.norm, dev)
        p_te = prediksi(model, tt, torch.arange(len(dte["meta"]), device=dev))
        np.save(os.path.join(run_dir, "test_probs.npy"), p_te.astype(np.float32))
    simpan_metrics(run_dir, m)
    print(f"[{args.nama}] macroF1 {m['macro_f1']:.4f} | bobot-test {m['macro_f1_bobot_test']:.4f} | "
          f"per lini {m['macro_f1_per_grup']} | tanpa sensor {m['macro_f1_tanpa_sensor']:.4f} | {m['waktu_detik']}s")
    return m


if __name__ == "__main__":
    main()
