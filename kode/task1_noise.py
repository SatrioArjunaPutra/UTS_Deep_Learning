"""Task 1d — deteksi kemungkinan salah label dengan prediksi out-of-fold (OOF).

Cara kerja:
1. Latih model fusion 5 kali (5-fold stratified). Setiap sampel latih mendapat probabilitas
   dari model yang TIDAK melihat sampel itu saat training (out-of-fold).
2. Sampel ditandai "kemungkinan salah label" bila probabilitas untuk label inspektur sangat
   rendah (< --ambang_label) DAN model yakin pada kelas lain (maks prob > --ambang_lain).
   Ide ini mirip confident learning (Northcutt dkk., 2021): label yang bertentangan dengan
   prediksi OOF yang yakin adalah kandidat salah label.
3. Simpan daftar id_unit, matriks label-inspektur x kelas-prediksi, dan contoh gambar.

Teknik yang tahan noise dibandingkan di task1_eksperimen.py --grup noise:
  CE biasa vs label smoothing 0.1 vs membuang sampel yang ditandai.

    python kode/task1_noise.py --data data_uts_dl
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, buat_run, folder_figur, simpan_metrics  # noqa: E402
from metrik import KELAS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--ambang_label", type=float, default=0.05)
    ap.add_argument("--ambang_lain", type=float, default=0.80)
    ap.add_argument("--lewati_training", action="store_true", help="pakai run OOF yang sudah ada")
    ap.add_argument("--seed", type=int, default=0, help="seed training tiap fold OOF (diteruskan ke task1_train.py)")
    a = ap.parse_args()
    run = buat_run("t1_noise", a)

    for f in range(a.k):
        nama = f"t1_oof_f{f}"
        if a.lewati_training and os.path.exists(os.path.join(ROOT, "runs", nama, "val_probs.npy")):
            continue
        subprocess.run([sys.executable, os.path.join(HERE, "task1_train.py"), "--data", a.data, "--nama", nama,
                        "--modal", "fusion", "--fusion", a.fusion, "--kfold", str(a.k), "--fold", str(f),
                        "--epochs", str(a.epochs), "--mod_drop", "0.1", "--seed", str(a.seed)], check=True)

    meta = pd.read_csv(os.path.join(a.data, "klasifikasi_train_meta.csv"))
    y = meta["kelas"].map({k: i for i, k in enumerate(KELAS)}).values
    oof = np.full((len(meta), len(KELAS)), np.nan, np.float32)
    for f in range(a.k):
        d = os.path.join(ROOT, "runs", f"t1_oof_f{f}")
        oof[np.load(os.path.join(d, "val_idx.npy"))] = np.load(os.path.join(d, "val_probs.npy"))
    assert not np.isnan(oof).any()
    np.save(os.path.join(run, "oof_probs.npy"), oof)

    p_label = oof[np.arange(len(y)), y]
    pred = oof.argmax(1)
    tandai = (p_label < a.ambang_label) & (oof.max(1) > a.ambang_lain)
    meta["p_label"] = p_label
    meta["prediksi_oof"] = np.array(KELAS)[pred]
    meta["p_prediksi"] = oof.max(1)
    flag = meta[tandai].sort_values("p_label")
    flag[["id_unit", "lini", "kelas", "prediksi_oof", "p_label", "p_prediksi"]].to_csv(
        os.path.join(run, "mungkin_salah_label.csv"), index=False)

    ct = pd.crosstab(flag["kelas"], flag["prediksi_oof"]).reindex(index=KELAS, columns=KELAS, fill_value=0)
    ct.to_csv(os.path.join(run, "matriks_salah_label.csv"))
    from metrik import macro_f1

    hasil = {
        "n_ditandai": int(tandai.sum()), "frac_ditandai": float(tandai.mean()),
        "ditandai_per_kelas_label": flag["kelas"].value_counts().to_dict(),
        "ditandai_per_lini": flag["lini"].value_counts().to_dict(),
        "pasangan_terbanyak": {f"{i}->{j}": int(ct.loc[i, j]) for i, j in
                               ct.stack().sort_values(ascending=False).head(8).index},
        "oof_macro_f1": macro_f1(y, pred),
        "oof_akurasi": float((pred == y).mean()),
        "ambang": {"p_label <": a.ambang_label, "p_maks >": a.ambang_lain},
        "frac_p_label_kurang_0.5": float((p_label < 0.5).mean()),
    }
    simpan_metrics(run, hasil)

    # contoh gambar yang ditandai
    g = np.load(os.path.join(a.data, "klasifikasi_train.npz"))["gambar"]
    contoh = flag.head(24)
    fig, ax = plt.subplots(3, 8, figsize=(16, 7))
    for axi, (_, r) in zip(ax.ravel(), contoh.iterrows()):
        axi.imshow(g[r["indeks"]], cmap="gray", vmin=0, vmax=255)
        axi.set_title(f"label: {r['kelas']}\nmodel: {r['prediksi_oof']} ({r['p_prediksi']:.2f})", fontsize=7)
        axi.axis("off")
    for axi in ax.ravel()[len(contoh):]:
        axi.axis("off")
    fig.suptitle("Kandidat salah label (prediksi out-of-fold sangat yakin pada kelas lain)")
    fig.tight_layout(); fig.savefig(os.path.join(folder_figur(1), "noise_contoh.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ct.values, cmap="Reds")
    ax.set_xticks(range(7)); ax.set_xticklabels(KELAS, rotation=45, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(KELAS)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xlabel("prediksi OOF"); ax.set_ylabel("label inspektur")
    ax.set_title(f"{int(tandai.sum())} kandidat salah label ({tandai.mean():.1%})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(folder_figur(1), "noise_matriks.png"), dpi=110)
    print(hasil)


if __name__ == "__main__":
    main()
