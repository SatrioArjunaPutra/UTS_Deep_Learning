"""Kumpulkan angka dari runs/*/metrics.json menjadi docs/tabel_laporan.md (P4.2).

Tidak ada angka yang ditulis tangan: semua dibaca dari file hasil run. Run yang belum ada
ditandai "BELUM ADA".

    python kode/tabel_laporan.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, baca_json  # noqa: E402
from metrik import KELAS  # noqa: E402

RUNS = os.path.join(ROOT, "runs")


def m(nama):
    p = os.path.join(RUNS, nama, "metrics.json")
    return baca_json(p) if os.path.exists(p) else None


def f(x, d=4):
    if x is None:
        return "–"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(v) if not isinstance(v, str) else v for v in r.values) + " |")
    return "\n".join(out)


def csv_md(path, cols=None, rename=None):
    if not os.path.exists(path):
        return f"_BELUM ADA: {os.path.relpath(path, ROOT)}_"
    df = pd.read_csv(path)
    if cols:
        df = df[[c for c in cols if c in df]]
    if rename:
        df = df.rename(columns=rename)
    return md(df)


def main():
    L = ["# Tabel angka laporan (dibuat otomatis oleh `kode/tabel_laporan.py`)", "",
         "Setiap angka berasal dari `runs/<run>/metrics.json` atau CSV di `runs/*_tabel/`.", ""]

    # ---------------- Task 1
    L += ["## Task 1 — Klasifikasi multimodal", ""]
    s = m("t1_sanity_overfit")
    L += [f"**Sanity check overfit 1 batch (64 sampel):** loss akhir {f(s and s['overfit_loss_akhir'])}, "
          f"akurasi akhir {f(s and s['overfit_acc_akhir'])} setelah {s and s['langkah']} langkah." if s else
          "**Sanity check:** BELUM ADA", ""]
    kol = ["run", "macro_f1", "macro_f1_lini_A", "macro_f1_lini_B", "macro_f1_bobot_test", "macro_f1_tanpa_sensor", "waktu_s"]
    for g, judul in [("modality", "Modality & fusion (validasi 9.600 unit)"), ("noise", "Label noise"),
                     ("domain", "Domain shift"), ("missing", "Missing modality")]:
        L += [f"### {judul}", "", csv_md(os.path.join(RUNS, "t1_tabel", f"{g}.csv"), kol + (["macro_f1_sensor_hilang_asli"] if g == "missing" else [])), ""]
    L += ["### F1 per kelas per modality", "",
          csv_md(os.path.join(RUNS, "t1_tabel", "modality.csv"), ["run"] + [f"f1_{k}" for k in KELAS]), ""]
    n = m("t1_noise")
    if n:
        L += [f"**Kandidat salah label (OOF 5-fold):** {n['n_ditandai']} unit ({n['frac_ditandai']:.2%}); "
              f"macro-F1 OOF {f(n['oof_macro_f1'])}. Pasangan terbanyak: {n['pasangan_terbanyak']}.", ""]
    p = os.path.join(RUNS, "t1_tabel", "final_mean_std.csv")
    L += ["### Model final (3 seed, mean ± std)", ""]
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0)
        L += ["| metrik | mean | std |", "|---|---|---|"] + [f"| {i} | {f(r['mean'])} | {f(r['std'])} |" for i, r in df.iterrows()]
    else:
        L += ["_BELUM ADA_"]
    L += [""]

    # ---------------- Task 2
    L += ["## Task 2 — RUL", ""]
    e = m("t2_eda")
    if e:
        L += [f"Sensor dibuang: {', '.join(e['sensor_tak_informatif'])}. Spike diganti {e['preprocessing_train']['n_spike_diganti']}, "
              f"NaN diisi {e['preprocessing_train']['n_nan_diisi']} (train).", ""]
    L += ["### Perbandingan model (5-fold per mesin, 5 titik potong acak per mesin validasi)", "",
          csv_md(os.path.join(RUNS, "t2_tabel", "model.csv"),
                 ["run", "rmse", "skor_total", "skor_per_titik", "rmse_fold_mean", "rmse_fold_std",
                  "rmse_rul_0_50", "skor_rul_0_50", "rmse_rul_50_100", "skor_rul_50_100", "rmse_rul_100_150",
                  "skor_rul_100_150", "rmse_semua_titik"]), ""]
    for u in sorted(glob.glob(os.path.join(RUNS, "t2_uncert_*", "metrics.json"))):
        x = baca_json(u)
        L += [f"### Ketidakpastian ({os.path.basename(os.path.dirname(u))}, fold 0, {x['n_titik_potong']} titik)", "",
              "| metrik | nilai |", "|---|---|"]
        for k in ["rmse_seed0", "rmse_ensemble", "skor_seed0", "skor_ensemble", "cakupan_mentah_mc_dropout_seed0",
                  "lebar_mentah_mc_dropout_seed0", "cakupan_mentah_ensemble_saja", "cakupan_mentah_ensemble+mc",
                  "lebar_mentah_ensemble+mc", "cakupan_conformal_split", "lebar_conformal_split", "faktor_conformal_q",
                  "korelasi_sd_vs_galat_abs"]:
            L.append(f"| {k} | {f(x.get(k))} |")
        L += ["", "Cakupan per rentang: " + str({k: round(v, 3) for k, v in x["cakupan_per_rentang"].items()}), "",
              md(pd.DataFrame(x["aturan_servis"])), "", f"Geser bias: {x['geser_bias']}", ""]
    t = m("t2_final")
    if t:
        L += [f"**Prediksi final:** {t['n_mesin']} mesin, RUL rata-rata {f(t['rul_rata2'], 2)}, "
              f"{t['frac_servis_sekarang']:.1%} direkomendasikan servis sekarang (batas bawah 90% ≤ 30).", ""]

    # ---------------- Task 3
    L += ["## Task 3 — Clustering", ""]
    b = m("t3_cluster_banding")
    if b:
        L += ["### Perbandingan representasi (tanpa label)", "", md(pd.DataFrame(b["tabel"])), ""]
    k = m("t3_cluster_k")
    if k:
        L += ["### Pemilihan jumlah cluster", "", f"Pilihan per kriteria: {k['pilihan_k']}", "",
              md(pd.DataFrame(k["tabel"])), ""]
    c = m("t3_cluster_final")
    if c:
        L += [f"### Final (k = {c['k']}, metode {c['metode_final']})", "", "| metode | silhouette | DB | NMI mikroskop | stabilitas 3 seed |",
              "|---|---|---|---|---|"]
        for mm, v in c["metode"].items():
            L.append(f"| {mm} | {f(v['silhouette_fusion'])} | {f(v['davies_bouldin'])} | {f(v['nmi_mikroskop'])} | {f(v['stabilitas_3seed'])} |")
        L += ["", f"ARI antar metode: {c['ARI_antar_metode']}", "", f"Modality: {c['modality']}", "",
              f"Cluster sulit: {c['cluster_sulit']}", ""]

    figs = sorted(glob.glob(os.path.join(ROOT, "laporan_figs", "*", "*.png")))
    L += ["## Daftar figur", ""] + [f"- `{os.path.relpath(p, ROOT)}`" for p in figs]
    out = os.path.join(ROOT, "docs", "tabel_laporan.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"ditulis: {out}")


if __name__ == "__main__":
    main()
