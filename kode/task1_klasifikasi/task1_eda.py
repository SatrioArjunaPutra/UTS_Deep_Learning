"""Task 1a — eksplorasi data klasifikasi multimodal.

Output:
- laporan_figs/task1/eda_contoh_gambar.png    : 6 contoh gambar per kelas (lini A & B)
- laporan_figs/task1/eda_sinyal_per_kelas.png : rata-rata sinyal tiap kanal per kelas, dan
                                                selisihnya terhadap kelas normal
- laporan_figs/task1/eda_distribusi.png       : distribusi kelas, sensor hilang per kelas/lini
- laporan_figs/task1/eda_shift_lini.png       : perbedaan train vs test per lini
- runs/t1_eda/metrics.json                    : semua angka EDA yang dikutip di laporan

    python kode/task1_klasifikasi/task1_eda.py --data data_uts_dl
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# folder task ini (modul task) + kode/ (paket common) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import buat_run, folder_figur, simpan_metrics  # noqa: E402
from metrik import KELAS  # noqa: E402

KANAL = ["suhu (C)", "getaran", "arus (A)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_uts_dl")
    a = ap.parse_args()
    fig_dir = folder_figur(1)
    run = buat_run("t1_eda", a)

    def muat(split):
        z = np.load(os.path.join(a.data, f"klasifikasi_{split}.npz"))
        return z["gambar"], z["sensor"].astype(np.float32), pd.read_csv(os.path.join(a.data, f"klasifikasi_{split}_meta.csv"))

    g, s, m = muat("train")
    gt, st, mt = muat("test")
    hasil = {"n_train": len(m), "n_test": len(mt)}

    # ---- distribusi kelas & sensor hilang
    hasil["distribusi_kelas"] = m["kelas"].value_counts().to_dict()
    kanal_hilang = np.isnan(s).all(axis=2)              # (N,3)
    kanal_hilang_t = np.isnan(st).all(axis=2)
    hasil["sensor_hilang_sebagian_dalam_kanal"] = float((np.isnan(s).any(2) & ~np.isnan(s).all(2)).mean())
    hasil["frac_kanal_hilang_train"] = dict(zip(KANAL, kanal_hilang.mean(0).round(4).tolist()))
    hasil["frac_semua_sensor_hilang_train"] = float(kanal_hilang.all(1).mean())
    hasil["frac_semua_sensor_hilang_test"] = float(kanal_hilang_t.all(1).mean())
    hasil["frac_min1_kanal_hilang_train"] = float(kanal_hilang.any(1).mean())
    m["semua_hilang"] = kanal_hilang.all(1)
    m["ada_hilang"] = kanal_hilang.any(1)
    hasil["sensor_hilang_per_kelas"] = m.groupby("kelas")["semua_hilang"].mean().round(4).to_dict()
    hasil["sensor_hilang_per_lini_train"] = m.groupby("lini")["semua_hilang"].mean().round(4).to_dict()
    mt["semua_hilang"] = kanal_hilang_t.all(1)
    hasil["sensor_hilang_per_lini_test"] = mt.groupby("lini")["semua_hilang"].mean().round(4).to_dict()

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    vc = m["kelas"].value_counts().reindex(KELAS)
    ax[0].bar(KELAS, vc.values, color="#4C72B0"); ax[0].set_title("Distribusi kelas (train)")
    ax[0].tick_params(axis="x", rotation=40)
    for i, v in enumerate(vc.values):
        ax[0].text(i, v, f"{v / len(m):.0%}", ha="center", va="bottom", fontsize=8)
    hk = m.groupby("kelas")[["semua_hilang", "ada_hilang"]].mean().reindex(KELAS) * 100
    hk.plot.bar(ax=ax[1], color=["#C44E52", "#DD8452"]); ax[1].set_title("% sensor hilang per kelas")
    ax[1].legend(["semua kanal", ">=1 kanal"], fontsize=8); ax[1].tick_params(axis="x", rotation=40)
    pl = pd.DataFrame({"train": m.groupby("lini")["semua_hilang"].mean(), "test": mt.groupby("lini")["semua_hilang"].mean()}) * 100
    pl.plot.bar(ax=ax[2], color=["#4C72B0", "#55A868"]); ax[2].set_title("% semua sensor hilang per lini")
    ax[2].tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_distribusi.png"), dpi=120); plt.close(fig)

    # ---- contoh gambar per kelas
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(len(KELAS), 6, figsize=(9, 11))
    for i, k in enumerate(KELAS):
        idx = list(rng.choice(np.where((m.kelas == k) & (m.lini == "A"))[0], 3, replace=False)) + \
              list(rng.choice(np.where((m.kelas == k) & (m.lini == "B"))[0], 3, replace=False))
        for j, ii in enumerate(idx):
            ax[i, j].imshow(g[ii], cmap="gray", vmin=0, vmax=255)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(f"lini {m.lini[ii]}", fontsize=7)
        ax[i, 0].set_ylabel(k, fontsize=9)
    fig.suptitle("Contoh gambar per kelas (3 lini A, 3 lini B)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_contoh_gambar.png"), dpi=110); plt.close(fig)

    # ---- sinyal per kelas
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ada = ~np.isnan(s).all(2)
    mu_normal = [np.nanmean(s[(m.kelas == "normal").values & ada[:, c], c], 0) for c in range(3)]
    beda = {}
    for k in KELAS:
        for c in range(3):
            sel = (m.kelas == k).values & ada[:, c]
            mu = np.nanmean(s[sel, c], 0)
            ax[0, c].plot(mu, label=k, lw=1)
            d = mu - mu_normal[c]
            ax[1, c].plot(d, label=k, lw=1)
            # besar beda rata-rata relatif terhadap std antar-sampel kelas normal
            sd = np.nanstd(s[(m.kelas == "normal").values & ada[:, c], c], 0).mean()
            beda.setdefault(k, {})[KANAL[c]] = round(float(np.abs(d).mean() / sd), 3)
    for c in range(3):
        ax[0, c].set_title(f"rata-rata {KANAL[c]}"); ax[1, c].set_title(f"selisih vs normal: {KANAL[c]}")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_sinyal_per_kelas.png"), dpi=110); plt.close(fig)
    hasil["beda_sinyal_vs_normal_relatif_std"] = beda

    # varians getaran (kanal 1) per kelas — sinyal getaran berosilasi, jadi bedanya bisa di energi
    var1 = np.nanstd(s[:, 1], axis=1)
    hasil["std_getaran_per_kelas"] = pd.Series(var1).groupby(m.kelas).mean().round(4).to_dict()

    # ---- train vs test per lini
    hasil["proporsi_lini"] = {"train": m.lini.value_counts(normalize=True).round(4).to_dict(),
                              "test": mt.lini.value_counts(normalize=True).round(4).to_dict()}
    hasil["proporsi_mesin_test_vs_train"] = (mt.id_mesin.value_counts(normalize=True) /
                                             m.id_mesin.value_counts(normalize=True)).round(2).to_dict()
    hasil["mesin_per_lini"] = m.groupby("lini").id_mesin.unique().apply(sorted).to_dict()
    gm, gmt = g.reshape(len(g), -1).mean(1), gt.reshape(len(gt), -1).mean(1)
    gs, gst = g.reshape(len(g), -1).std(1), gt.reshape(len(gt), -1).std(1)
    hasil["kecerahan_rata2"] = {f"{sp}_{l}": round(float(x[mm.lini == l].mean()), 2)
                                for sp, x, mm in [("train", gm, m), ("test", gmt, mt)] for l in "AB"}
    hasil["kontras_rata2"] = {f"{sp}_{l}": round(float(x[mm.lini == l].mean()), 2)
                              for sp, x, mm in [("train", gs, m), ("test", gst, mt)] for l in "AB"}
    hasil["suhu_rata2_per_lini"] = {l: round(float(np.nanmean(s[(m.lini == l).values, 0])), 3) for l in "AB"}
    hasil["kelas_per_lini"] = pd.crosstab(m.kelas, m.lini, normalize="columns").round(4).to_dict()

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    pd.DataFrame(hasil["proporsi_lini"]).plot.bar(ax=ax[0], color=["#4C72B0", "#55A868"])
    ax[0].set_title("Proporsi lini: train vs test"); ax[0].tick_params(axis="x", rotation=0)
    for l, c in zip("AB", ["#4C72B0", "#C44E52"]):
        ax[1].hist(gm[m.lini == l], bins=60, alpha=.5, density=True, color=c, label=f"train {l}")
        ax[1].hist(gmt[mt.lini == l], bins=60, histtype="step", density=True, color=c, label=f"test {l}")
        ax[2].hist(np.nanmean(s[(m.lini == l).values, 0], 1), bins=60, alpha=.5, density=True, color=c, label=l)
    ax[1].set_title("Kecerahan rata-rata gambar"); ax[1].legend(fontsize=7)
    ax[2].set_title("Suhu rata-rata per unit (train)"); ax[2].legend(fontsize=7)
    ct = pd.crosstab(m.kelas, m.lini, normalize="columns").reindex(KELAS) * 100
    ct.plot.bar(ax=ax[3]); ax[3].set_title("% kelas per lini (train)"); ax[3].tick_params(axis="x", rotation=40)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_shift_lini.png"), dpi=110); plt.close(fig)

    simpan_metrics(run, hasil)
    import json
    print(json.dumps(hasil, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
