"""Task 1 — menjalankan grup eksperimen dan membuat tabel/figur ringkasan.

Grup (--grup):
  sanity   : overfit satu batch (fusion)
  modality : image-only, sensor-only, fusion concat / late / xattn          (poin 1b, 1c)
  noise    : baseline vs label smoothing 0.1 vs buang kandidat salah label  (poin 1d)
  domain   : baseline vs normalisasi per lini vs augmentasi vs bobot lini   (poin 1e)
  missing  : baseline vs modality dropout 0.25 (+ evaluasi tanpa sensor)    (poin 1f)
  final    : konfigurasi final, 3 seed, + prediksi data uji                 (poin 1g)
  tabel    : hanya membuat tabel/figur dari run yang sudah ada

Konfigurasi dasar setiap grup bisa diubah lewat --fusion/--extra agar grup berikutnya
memakai pilihan terbaik dari grup sebelumnya. Hasil tabel: runs/t1_tabel/*.csv dan
laporan_figs/task1/*.png.

    python kode/task1_klasifikasi/task1_eksperimen.py --grup modality
"""
import argparse
import os
import shlex
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# folder task ini (modul task) + kode/ (paket common) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import ROOT, baca_json, config_beda, folder_figur  # noqa: E402
from metrik import KELAS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
DATA_DIR = ["data_uts_dl"]   # diisi dari --data di main()


def jalankan(nama, argumen, data, ulang=False):
    if not ulang and os.path.exists(os.path.join(RUNS, nama, "metrics.json")):
        beda = config_beda(os.path.join(RUNS, nama), argumen)
        if not beda:
            print(f"[lewati] {nama} sudah ada")
            return
        print(f"[ULANG] {nama} ada tetapi konfigurasinya berbeda: {beda}", flush=True)
    cmd = [sys.executable, os.path.join(HERE, "task1_train.py"), "--data", data, "--nama", nama] + argumen
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    # Di Windows proses CUDA kadang crash SAAT KELUAR (0xC0000409) setelah semua file tersimpan
    if not os.path.exists(os.path.join(RUNS, nama, "metrics.json")):
        raise RuntimeError(f"{nama} gagal (exit {r.returncode})")


def grup_runs(fusion, extra):
    base = ["--modal", "fusion", "--fusion", fusion] + extra
    tag = fusion + ("_" + "_".join(a.strip("-") for a in extra if a.startswith("--")) if extra else "")
    # baseline tiap grup = run fusion yang sama di grup modality (konfigurasi identik, tidak dilatih ulang)
    nama_base = f"t1_mod_{fusion}" if not extra else f"t1_base_{tag}"
    return {
        "sanity": {"t1_sanity_overfit": ["--overfit_batch", "--modal", "fusion", "--fusion", fusion]},
        "modality": {
            "t1_mod_image": ["--modal", "image"],
            "t1_mod_sensor": ["--modal", "sensor"],
            "t1_mod_concat": ["--modal", "fusion", "--fusion", "concat"],
            "t1_mod_late": ["--modal", "fusion", "--fusion", "late"],
            "t1_mod_xattn": ["--modal", "fusion", "--fusion", "xattn"],
        },
        "noise": {
            nama_base: base,
            f"t1_noise_ls_{tag}": base + ["--ls", "0.1"],
            # path relatif terhadap root repo (task1_train menyelesaikannya) -> sama di komputer mana pun
            f"t1_noise_buang_{tag}": base + ["--drop_ids", "runs/t1_noise/mungkin_salah_label.csv"],
        },
        "domain": {
            nama_base: base,
            f"t1_dom_perline_{tag}": base + ["--norm", "perline"],
            f"t1_dom_aug_{tag}": base + ["--aug"],
            f"t1_dom_reweight_{tag}": base + ["--line_reweight"],
            f"t1_dom_perline_aug_{tag}": base + ["--norm", "perline", "--aug"],
            # garis halus berarah lini B, arah acak per gambar (lihat analisis Fourier di task1_train.py)
            f"t1_dom_tekstur_{tag}": base + ["--aug_tekstur", "0.5"],
        },
        "missing": {
            nama_base: base,
            f"t1_miss_drop25_{tag}": base + ["--mod_drop", "0.25"],
        },
        "final": {f"t1_final_s{s}": base + ["--seed", str(s), "--save_test"] for s in range(3)},
    }


def baris(nama):
    p = os.path.join(RUNS, nama, "metrics.json")
    if not os.path.exists(p):
        return None
    m = baca_json(p)
    r = {"run": nama, "macro_f1": m.get("macro_f1"), "macro_f1_lini_A": m.get("macro_f1_per_grup", {}).get("A"),
         "macro_f1_lini_B": m.get("macro_f1_per_grup", {}).get("B"), "macro_f1_bobot_test": m.get("macro_f1_bobot_test"),
         "macro_f1_tanpa_sensor": m.get("macro_f1_tanpa_sensor"),
         "macro_f1_sensor_hilang_asli": m.get("macro_f1_sensor_hilang_asli"), "waktu_s": m.get("waktu_detik")}
    for k, v in (m.get("f1_per_kelas") or {}).items():
        r[f"f1_{k}"] = v
    return r


def tabel(runs_per_grup):
    out = os.path.join(RUNS, "t1_tabel")
    os.makedirs(out, exist_ok=True)
    fig_dir = folder_figur(1)
    semua = {}
    for g, runs in runs_per_grup.items():
        if g == "sanity":
            continue
        rows = [r for r in (baris(n) for n in runs) if r]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"{g}.csv"), index=False)
        semua[g] = df
        print(f"\n=== {g}\n" + df.drop(columns=[c for c in df if c.startswith("f1_")]).round(4).to_string(index=False))

    # F1 per kelas per lini untuk semua run grup modality/domain/noise/missing (dari prediksi validasi
    # tersimpan runs/<run>/val_probs.npy; run lama tidak punya kunci ini di metrics.json)
    from metrik import f1_per_kelas
    meta = pd.read_csv(os.path.join(DATA_DIR[0], "klasifikasi_train_meta.csv"))
    y_all = meta["kelas"].map({k: i for i, k in enumerate(KELAS)}).values
    lini_all = meta["lini"].values
    rows = []
    for g in ["modality", "noise", "domain", "missing", "final"]:
        for n in runs_per_grup.get(g, {}):
            d = os.path.join(RUNS, n)
            if not os.path.exists(os.path.join(d, "val_probs.npy")):
                continue
            vi = np.load(os.path.join(d, "val_idx.npy"))
            pv = np.load(os.path.join(d, "val_probs.npy")).argmax(1)
            for l in ["A", "B"]:
                mk = lini_all[vi] == l
                rows.append({"run": n, "lini": l, **dict(zip(KELAS, f1_per_kelas(y_all[vi][mk], pv[mk])))})
    if rows:
        # gabung dengan baris lama (pemanggilan dengan --extra hanya melihat sebagian run), baris baru menang
        p_csv = os.path.join(out, "f1_per_kelas_per_lini.csv")
        lama = pd.read_csv(p_csv) if os.path.exists(p_csv) else pd.DataFrame()
        pd.concat([lama, pd.DataFrame(rows)]).drop_duplicates(["run", "lini"], keep="last").round(4).to_csv(
            p_csv, index=False)

    if "modality" in semua:  # heatmap F1 per kelas per modality
        df = semua["modality"].set_index("run")
        h = df[[f"f1_{k}" for k in KELAS]]
        h.columns = KELAS
        h.index = [i.replace("t1_mod_", "") for i in h.index]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        im = ax.imshow(h.values.astype(float), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_xticklabels(KELAS, rotation=30, ha="right")
        ax.set_yticks(range(len(h))); ax.set_yticklabels(h.index)
        for i in range(len(h)):
            for j in range(7):
                v = h.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="w" if v < .6 else "k")
        ax.set_title("F1 per kelas per modality (validasi)")
        fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "modality_f1_heatmap.png"), dpi=120)
        plt.close(fig)

    if "final" in semua:
        df = semua["final"]
        cols = ["macro_f1", "macro_f1_lini_A", "macro_f1_lini_B", "macro_f1_bobot_test", "macro_f1_tanpa_sensor"] + \
               [f"f1_{k}" for k in KELAS]
        ms = pd.DataFrame({"mean": df[cols].mean(), "std": df[cols].std(ddof=1), "n_seed": len(df)})
        ms.to_csv(os.path.join(out, "final_mean_std.csv"))
        print("\n=== final mean ± std\n" + ms.round(4).to_string())
        # learning curves 3 seed + confusion matrix (dijumlah atas seed)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        cm = 0
        for n in df["run"]:
            h = baca_json(os.path.join(RUNS, n, "history.json"))
            e = [x["epoch"] for x in h]
            ax[0].plot(e, [x["train_loss"] for x in h], "--", label=f"{n[-2:]} train")
            ax[0].plot(e, [x["val_loss"] for x in h], label=f"{n[-2:]} val")
            ax[1].plot(e, [x["val_f1"] for x in h], label=n[-2:])
            cm = cm + np.load(os.path.join(RUNS, n, "confusion.npy"))
        ax[0].set_title("Loss (train putus-putus, validasi garis)"); ax[0].legend(fontsize=7); ax[0].set_xlabel("epoch")
        ax[1].set_title("Macro-F1 validasi"); ax[1].legend(fontsize=7); ax[1].set_xlabel("epoch")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "final_kurva.png"), dpi=120); plt.close(fig)
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_xticklabels(KELAS, rotation=45, ha="right")
        ax.set_yticks(range(7)); ax.set_yticklabels(KELAS)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="w" if cmn[i, j] > .5 else "k")
        ax.set_xlabel("prediksi"); ax.set_ylabel("label (validasi)"); ax.set_title("Confusion matrix final (3 seed)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "final_confusion.png"), dpi=120); plt.close(fig)
    return semua


def buat_prediksi(data):
    """Ensemble (rata-rata probabilitas) 3 seed final -> prediksi_klasifikasi.csv di root repo."""
    meta = pd.read_csv(os.path.join(data, "klasifikasi_test_meta.csv"))
    probs = [np.load(os.path.join(RUNS, f"t1_final_s{s}", "test_probs.npy")) for s in range(3)]
    p = np.mean(probs, 0)
    out = pd.DataFrame({"id_unit": meta["id_unit"], "kelas": np.array(KELAS)[p.argmax(1)]})
    out.to_csv(os.path.join(ROOT, "prediksi_klasifikasi.csv"), index=False)
    print("prediksi_klasifikasi.csv:", out["kelas"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--grup", required=True,
                    choices=["sanity", "modality", "noise", "domain", "missing", "final", "tabel", "prediksi"])
    ap.add_argument("--fusion", default="concat", help="fusion dasar untuk grup noise/domain/missing/final")
    ap.add_argument("--extra", default="", help="argumen tambahan konfigurasi dasar, mis. \"--mod_drop 0.25\"")
    ap.add_argument("--ulang", action="store_true")
    a = ap.parse_args()
    extra = shlex.split(a.extra)
    DATA_DIR[0] = a.data
    rg = grup_runs(a.fusion, extra)
    if a.grup == "prediksi":
        return buat_prediksi(a.data)
    if a.grup != "tabel":
        for nama, arg in rg[a.grup].items():
            jalankan(nama, arg, a.data, a.ulang)
        if a.grup == "final":
            buat_prediksi(a.data)
    # tabel memakai semua run yang ada (nama run dengan konfigurasi dasar saat ini)
    tabel(rg)


if __name__ == "__main__":
    main()
