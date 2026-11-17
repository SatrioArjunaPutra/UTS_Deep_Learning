"""Task 3a — eksplorasi distorsi gambar mikrostruktur dan spektrum komposisi.

Diukur tanpa label:
- Gambar: kecerahan & kontras per mikroskop (M1/M2), orientasi dominan dan frekuensi spasial
  dominan dari spektrum Fourier 2D (proksi rotasi & skala).
- Spektrum: posisi puncak (proksi pergeseran), kemiringan baseline (regresi pada persentil
  bawah), jumlah spike (titik tunggal yang jauh di atas median-filter tetangganya).

Output: laporan_figs/task3/eda_*.png dan runs/t3_eda/metrics.json

    python kode/task3_eda.py --data data_uts_dl
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import buat_run, folder_figur, simpan_metrics  # noqa: E402


def orientasi_frekuensi(g):
    """Orientasi (derajat, 0-180) & frekuensi radial dominan dari |FFT2| tiap gambar."""
    x = g.astype(np.float32)
    x = x - x.mean((1, 2), keepdims=True)
    F = np.abs(np.fft.fftshift(np.fft.fft2(x), axes=(1, 2)))
    n = g.shape[1]
    yy, xx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    r = np.sqrt(xx ** 2 + yy ** 2)
    F[:, r < 2] = 0                                   # buang komponen DC/sangat rendah
    k = F.reshape(len(g), -1).argmax(1)
    fy, fx = yy.ravel()[k], xx.ravel()[k]
    ang = (np.degrees(np.arctan2(fy, fx)) + 180) % 180
    # "kekuatan arah": rasio puncak terhadap rata-rata (tekstur bergaris -> tinggi)
    kuat = F.reshape(len(g), -1).max(1) / (F.reshape(len(g), -1).mean(1) + 1e-6)
    return ang, r.ravel()[k], kuat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_uts_dl")
    a = ap.parse_args()
    fig_dir = folder_figur(3)
    run = buat_run("t3_eda", a)
    z = np.load(os.path.join(a.data, "clustering.npz"))
    g, s = z["gambar"], z["spektrum"].astype(np.float32)
    m = pd.read_csv(os.path.join(a.data, "clustering_meta.csv"))
    mik = m["mikroskop"].values
    h = {"n": len(m), "mikroskop": m["mikroskop"].value_counts().to_dict()}

    # ---- gambar
    mean, std = g.reshape(len(g), -1).mean(1), g.reshape(len(g), -1).std(1)
    ang, frek, kuat = orientasi_frekuensi(g)
    h["kecerahan_per_mikroskop"] = {k: round(float(mean[mik == k].mean()), 2) for k in ["M1", "M2"]}
    h["kontras_per_mikroskop"] = {k: round(float(std[mik == k].mean()), 2) for k in ["M1", "M2"]}
    h["frek_dominan_per_mikroskop"] = {k: round(float(frek[mik == k].mean()), 2) for k in ["M1", "M2"]}
    h["orientasi_hist_10bin"] = np.histogram(ang[kuat > 20], bins=10, range=(0, 180))[0].tolist()
    h["frac_tekstur_berarah_kuat"] = float((kuat > 20).mean())
    # tajam/blur: energi Laplacian
    lap = np.abs(np.diff(g.astype(np.float32), 2, axis=1)).mean((1, 2))
    h["ketajaman_laplacian_per_mikroskop"] = {k: round(float(lap[mik == k].mean()), 2) for k in ["M1", "M2"]}

    # ---- spektrum
    mf = median_filter(s, size=(1, 5))
    resid = s - mf
    thr = 6 * np.median(np.abs(resid), axis=1, keepdims=True) * 1.4826 + 0.3
    spike = resid > thr
    h["spike_per_spektrum_rata2"] = float(spike.sum(1).mean())
    h["frac_spektrum_dengan_spike"] = float(spike.any(1).mean())
    t = np.arange(s.shape[1])
    base = np.array([np.polyfit(t[x <= np.percentile(x, 30)], x[x <= np.percentile(x, 30)], 1)[0] for x in mf])
    h["kemiringan_baseline_persentil"] = np.percentile(base * 128, [5, 25, 50, 75, 95]).round(3).tolist()
    pk = mf.argmax(1)
    h["posisi_puncak_tertinggi_persentil"] = np.percentile(pk, [5, 25, 50, 75, 95]).tolist()
    h["spektrum_min_max"] = [float(s.min()), float(s.max())]
    h["spike_per_mikroskop"] = {k: round(float(spike[mik == k].sum(1).mean()), 3) for k in ["M1", "M2"]}
    h["baseline_per_mikroskop"] = {k: round(float((base * 128)[mik == k].mean()), 3) for k in ["M1", "M2"]}

    # ---- figur contoh
    rng = np.random.RandomState(1)
    idx = rng.choice(len(g), 12, replace=False)
    fig, ax = plt.subplots(3, 12, figsize=(22, 6), gridspec_kw={"height_ratios": [1.2, 1, 1]})
    for j, i in enumerate(idx):
        ax[0, j].imshow(g[i], cmap="gray", vmin=0, vmax=255); ax[0, j].axis("off")
        ax[0, j].set_title(f"{mik[i]} θ={ang[i]:.0f}° f={frek[i]:.0f}", fontsize=7)
        ax[1, j].plot(s[i], lw=.8); ax[1, j].plot(np.where(spike[i])[0], s[i][spike[i]], "rx", ms=4)
        ax[1, j].set_xticks([]); ax[1, j].set_title(f"spike={spike[i].sum()}", fontsize=7)
        ax[2, j].plot(mf[i], lw=.8, color="g"); ax[2, j].set_xticks([])
    ax[1, 0].set_ylabel("spektrum"); ax[2, 0].set_ylabel("median filter")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_contoh.png"), dpi=100); plt.close(fig)

    fig, ax = plt.subplots(1, 5, figsize=(22, 3.6))
    for k, c in [("M1", "#4C72B0"), ("M2", "#C44E52")]:
        ax[0].hist(mean[mik == k], bins=60, alpha=.5, density=True, color=c, label=k)
        ax[1].hist(frek[mik == k], bins=np.arange(0, 33), alpha=.5, density=True, color=c, label=k)
    ax[0].set_title("Kecerahan rata-rata per mikroskop"); ax[0].legend()
    ax[1].set_title("Frekuensi spasial dominan (skala)"); ax[1].legend()
    ax[2].hist(ang[kuat > 20], bins=36); ax[2].set_title("Orientasi dominan (tekstur berarah)")
    ax[3].hist(pk, bins=64); ax[3].set_title("Posisi puncak tertinggi spektrum")
    ax[4].hist(base * 128, bins=60); ax[4].set_title("Kemiringan baseline (per 128 titik)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_distorsi.png"), dpi=110); plt.close(fig)

    simpan_metrics(run, h)
    import json
    print(json.dumps(h, indent=1, default=str))


if __name__ == "__main__":
    main()
