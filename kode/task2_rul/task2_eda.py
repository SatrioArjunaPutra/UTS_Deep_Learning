"""Task 2a — eksplorasi data RUL: regime operasi, sensor tak informatif, NaN, spike, panjang riwayat.

Output:
- laporan_figs/task2/eda_regime.png     : sensor mentah vs setel_1 (6 regime) & sebelum/sesudah normalisasi
- laporan_figs/task2/eda_sensor.png     : korelasi tiap sensor (dinormalisasi per regime) dengan RUL
- laporan_figs/task2/eda_panjang.png    : panjang riwayat train (sampai rusak) vs test (terpotong)
- laporan_figs/task2/eda_contoh_mesin.png : satu mesin sebelum/sesudah preprocessing
- runs/t2_eda/metrics.json

    python kode/task2_eda.py --data data_uts_dl
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# folder task ini (modul task) + kode/ (paket common, metrik) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import buat_run, folder_figur, simpan_metrics  # noqa: E402
from task2_prep import PUSAT_REGIME, SENSOR, SENSOR_PAKAI, Prep, label_rul, muat, regime  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_uts_dl")
    a = ap.parse_args()
    fig_dir = folder_figur(2)
    run = buat_run("t2_eda", a)
    tr, te = muat(a.data, "train"), muat(a.data, "test")
    tr["reg"], te["reg"] = regime(tr), regime(te)
    tr["rul"] = label_rul(tr)
    h = {"n_mesin_train": int(tr.id_unit.nunique()), "n_mesin_test": int(te.id_unit.nunique()),
         "n_baris_train": len(tr), "n_baris_test": len(te)}

    # regime
    h["jumlah_baris_per_regime"] = {str(int(PUSAT_REGIME[k])): int(v) for k, v in tr.reg.value_counts().sort_index().items()}
    h["setel_per_regime"] = tr.groupby("reg")[["setel_1", "setel_2", "setel_3"]].mean().round(3).to_dict()
    h["setel_std_dalam_regime"] = tr.groupby("reg")[["setel_1", "setel_2"]].std().max().round(4).to_dict()
    # rasio std global / std dalam regime
    rows = []
    for s in SENSOR:
        g = tr.groupby("reg")[s]
        z = (tr[s] - g.transform("mean")) / g.transform("std")
        rows.append({"sensor": s, "std_global": tr[s].std(), "std_dalam_regime": g.std().mean(),
                     "rasio": tr[s].std() / g.std().mean(), "korelasi_rul_mentah": tr[s].corr(tr.rul),
                     "korelasi_rul_norm_regime": z.corr(tr.rul), "frac_nan": tr[s].isna().mean(),
                     "n_unik": tr[s].nunique(), "dipakai": s in SENSOR_PAKAI})
    sens = pd.DataFrame(rows)
    sens.round(4).to_csv(os.path.join(run, "sensor.csv"), index=False)
    h["sensor_tak_informatif"] = sens.loc[sens.korelasi_rul_norm_regime.abs() < 0.02, "sensor"].tolist()
    h["frac_nan_per_sensor_rata2"] = float(sens.frac_nan.mean())
    h["frac_baris_ada_nan"] = float(tr[SENSOR].isna().any(axis=1).mean())

    # panjang riwayat
    Ltr, Lte = tr.groupby("id_unit").siklus.max(), te.groupby("id_unit").siklus.max()
    h["panjang_train"] = Ltr.describe().round(1).to_dict()
    h["panjang_test"] = Lte.describe().round(1).to_dict()
    h["siklus_berurutan_tanpa_lubang"] = bool((tr.groupby("id_unit").siklus.agg(lambda v: (np.diff(v) == 1).all())).all())

    # spike & NaN yang ditangani preprocessing
    _, info = Prep().fit(tr).transform(tr, kembalikan_info=True)
    h["preprocessing_train"] = info

    # ---- figur regime
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    smp = tr.sample(20000, random_state=0)
    ax[0].scatter(smp.setel_1, smp.setel_2, c=smp.setel_3, s=3, cmap="coolwarm")
    ax[0].set_xlabel("setel_1"); ax[0].set_ylabel("setel_2"); ax[0].set_title("Setelan operasi -> 6 regime (warna = setel_3)")
    ax[1].scatter(smp.setel_1 + np.random.RandomState(0).normal(0, .5, len(smp)), smp.s20, s=2, alpha=.4)
    ax[1].set_xlabel("setel_1 (+jitter)"); ax[1].set_ylabel("s20 mentah"); ax[1].set_title("Sensor mentah didominasi regime")
    u = tr[tr.id_unit == tr.id_unit.iloc[0]]
    z = Prep().fit(tr).transform(u)
    j = SENSOR_PAKAI.index("s04")
    ax[2].plot(u.siklus, (u.s04 - u.s04.mean()) / u.s04.std(), lw=.7, alpha=.6, label="s04 mentah (z global)")
    ax[2].plot(u.siklus, z[:, j], lw=.9, label="s04 dinormalisasi per regime")
    ax[2].set_xlabel("siklus"); ax[2].legend(fontsize=8); ax[2].set_title(f"Mesin {u.id_unit.iloc[0]}: tren degradasi terlihat")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_regime.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    c = ["#55A868" if d else "#C44E52" for d in sens.dipakai]
    ax.bar(sens.sensor, sens.korelasi_rul_norm_regime, color=c)
    ax.axhline(0, color="k", lw=.5)
    ax.set_title("Korelasi sensor (normalisasi per regime) dengan RUL — merah = dibuang")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_sensor.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(Ltr, bins=40, alpha=.6, label="train (sampai rusak)")
    ax.hist(Lte, bins=40, alpha=.6, label="test (terpotong)")
    ax.set_xlabel("panjang riwayat (siklus)"); ax.legend(); ax.set_title("Panjang riwayat per mesin")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_panjang.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for s in ["s04", "s12", "s17", "s02"]:
        ax[0].plot(u.siklus, u[s], lw=.7, label=s)
        ax[1].plot(u.siklus, z[:, SENSOR_PAKAI.index(s)], lw=.7, label=s)
    ax[0].set_title("Sebelum preprocessing (mentah, ada NaN)"); ax[0].legend(fontsize=8)
    ax[1].set_title("Sesudah: normalisasi per regime + spike kausal + forward-fill"); ax[1].set_xlabel("siklus")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "eda_contoh_mesin.png"), dpi=110); plt.close(fig)

    simpan_metrics(run, h)
    print(sens.round(3).to_string(index=False))
    import json
    print(json.dumps(h, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
