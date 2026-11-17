"""Task 1e — analisis Fourier perbedaan tekstur gambar lini A vs B (dasar augmentasi tekstur).

Hanya memakai gambar kelas `normal` dari DATA LATIH (tanpa data uji), supaya perbedaan yang diukur
murni karena lini, bukan karena cacat. Yang dihitung (runs/t1_tekstur/metrics.json):
- porsi energi pada cincin frekuensi 13-18 siklus/64 px (periode ±4 px) per gambar, lini A vs B;
- konsentrasi energi cincin pada satu arah (tinggi = garis lurus berarah, rendah = butiran tak berarah)
  dan histogram sudut puncak;
- energi absolut cincin (porsi × varians piksel) lini A, lini B, dan lini A + butiran tiruan
  (`butiran()` di task1_train.py: garis halus berarah, arah acak per gambar) — cek bahwa augmentasi meniru lini B;
- porsi energi cincin dalam ±20° dari arah puncak tiap gambar (garis berarah vs butiran tak berarah).
Figur: laporan_figs/task1/domain_fourier.png (log rasio spektrum daya rata-rata B / A).

    python kode/task1_tekstur.py --data data_uts_dl
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import buat_run, folder_figur, set_seed, simpan_metrics  # noqa: E402


def spektrum(x):
    x = x - x.mean((-2, -1), keepdims=True)
    return np.abs(np.fft.fftshift(np.fft.fft2(x), axes=(-2, -1))) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--n", type=int, default=2000, help="jumlah gambar normal per lini")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    run = buat_run("t1_tekstur", a)
    import torch

    from task1_train import TEKSTUR_FREK, TEKSTUR_STD_PX, butiran

    g = np.load(os.path.join(a.data, "klasifikasi_train.npz"))["gambar"]
    meta = pd.read_csv(os.path.join(a.data, "klasifikasi_train_meta.csv"))
    rng = np.random.RandomState(a.seed)
    idx = {l: rng.choice(np.where((meta.lini == l) & (meta.kelas == "normal"))[0], a.n, replace=False) for l in "AB"}
    yy, xx = np.mgrid[-32:32, -32:32]
    rad = np.hypot(yy, xx)
    cincin = (rad >= TEKSTUR_FREK[0]) & (rad <= TEKSTUR_FREK[1])

    def ukur(X):
        P = spektrum(X)
        pr = P[:, cincin]
        porsi = pr.sum(1) / P.reshape(len(P), -1).sum(1)
        konsentrasi = np.sort(pr, 1)[:, -2:].sum(1) / pr.sum(1)
        k = pr.argmax(1)
        sudut_c = np.degrees(np.arctan2(yy[cincin], xx[cincin])) % 180
        sudut = sudut_c[k]
        # porsi energi cincin dalam ±20° dari arah puncak gambar itu (garis berarah -> tinggi)
        d = np.abs((sudut_c[None] - sudut[:, None] + 90) % 180 - 90)
        dekat = (pr * (d <= 20)).sum(1) / pr.sum(1)
        var = X.reshape(len(X), -1).var(1)
        return {"porsi_energi_cincin_median": float(np.median(porsi)),
                "konsentrasi_satu_arah_median": float(np.median(konsentrasi)),
                "porsi_energi_cincin_dalam_20deg_puncak_median": float(np.median(dekat)),
                "hist_sudut_puncak_6bin_0_180": np.histogram(sudut, bins=6, range=(0, 180))[0].tolist(),
                "std_piksel_rata2": float(np.sqrt(var).mean()),
                "energi_cincin_absolut_median": float(np.median(porsi * var))}

    XA = g[idx["A"]].astype(np.float32)
    XB = g[idx["B"]].astype(np.float32)
    torch.manual_seed(a.seed)
    XA_aug = XA + TEKSTUR_STD_PX * butiran(len(XA), torch.device("cpu"))[:, 0].numpy()
    hasil = {"frek_cincin_siklus_per_64px": list(TEKSTUR_FREK), "std_butiran_px": TEKSTUR_STD_PX, "n_per_lini": a.n,
             "lini_A": ukur(XA), "lini_B": ukur(XB), "lini_A_plus_butiran": ukur(XA_aug)}
    simpan_metrics(run, hasil)

    r = np.log(spektrum(XB).mean(0) + 1) - np.log(spektrum(XA).mean(0) + 1)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    ax[0].imshow(g[idx["A"][0]], cmap="gray"); ax[0].set_title("normal lini A")
    ax[1].imshow(g[idx["B"][0]], cmap="gray"); ax[1].set_title("normal lini B")
    ax[2].imshow(np.clip(XA_aug[0], 0, 255), cmap="gray"); ax[2].set_title("lini A + garis tiruan (augmentasi)")
    im = ax[3].imshow(r, cmap="magma", extent=[-32, 32, 32, -32])
    ax[3].set_title("log rasio spektrum daya B / A\n(cincin = garis halus periode ±4 px)")
    ax[3].set_xlabel("frek x (siklus/64 px)"); ax[3].set_ylabel("frek y")
    for x in ax[:3]:
        x.axis("off")
    fig.colorbar(im, ax=ax[3]); fig.tight_layout()
    fig.savefig(os.path.join(folder_figur(1), "domain_fourier.png"), dpi=120); plt.close(fig)
    for k in ["lini_A", "lini_B", "lini_A_plus_butiran"]:
        print(k, hasil[k])


if __name__ == "__main__":
    main()
