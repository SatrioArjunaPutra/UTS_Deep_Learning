"""Satu perintah untuk mereproduksi angka laporan dan ketiga file prediksi.

    python kode/reproduce.py --data data_uts_dl --task all             # CEPAT: dari weights/ (< 5 menit, GPU)
    python kode/reproduce.py --data data_uts_dl --task 2 --retrain     # latih ulang Task 2 dari nol
    python kode/reproduce.py --data data_uts_dl --task all --retrain   # latih ulang SEMUA (beberapa jam)

Mode cepat (tanpa --retrain) memuat weights final di weights/ lalu menghitung ulang:
- Task 1: macro-F1 validasi tiap seed final (+ per lini, tanpa sensor), mean ± std, dan
          prediksi_klasifikasi.csv (ensemble 3 seed).
- Task 2: RMSE & skor asimetris validasi silang 5 fold model final (titik potong), dan
          prediksi_rul.csv (ensemble seed + MC dropout).
- Task 3: embedding dari encoder tersimpan -> clustering final -> cluster.csv + metrik internal.
Mode --retrain menjalankan seluruh pipeline (EDA + semua eksperimen di laporan), menyalin
weights final ke weights/, lalu menjalankan mode cepat. Setelah itu tabel laporan dibuat ulang
(docs/tabel_laporan.md).

Konfigurasi final (hasil pemilihan di validasi, lihat laporan) ada di FINAL di bawah.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, set_seed, simpan_json, versi_library  # noqa: E402

W = os.path.join(ROOT, "weights")
RUNS = os.path.join(ROOT, "runs")

# Konfigurasi final — diisi dari hasil eksperimen validasi (lihat laporan & docs/tabel_laporan.md)
# Task 1: xattn (terbaik di grup modality) + buang kandidat salah label (grup noise) + modality dropout 25%
#   (grup missing). Tidak ada teknik domain yang menaikkan macro-F1 tertimbang komposisi test -> tidak dipakai.
# Task 2: GRU, W=50 (grup model & window), resep hasil tuning t2_tune_gru_E (10 epoch).
# Task 3: representasi dipilih di t3_cluster_banding (ARI citra-vs-spektrum rata-rata k=6,10 tertinggi):
#   citra SimCLR-citra + spektrum SimCLR-joint; k=8 dari t3_cluster_k (uji belah bertingkat: 5 cluster kasar
#   silhouette/DB + 3 cluster yang terbelah di satu modality); metode = K-Means (aturan auto di t3_cluster_final:
#   silhouette seri dengan DEC, stabilitas antar seed lebih tinggi).
FINAL = {
    "t1_fusion": "xattn",
    "t1_extra": ["--mod_drop", "0.25", "--drop_ids", "runs/t1_noise/mungkin_salah_label.csv"],
    "t2_model": "gru",
    "t2_W": 50,
    "t2_epochs": 10,
    "t2_seed": 5,
    "t2_geser": 0.0,
    "t3_img": "runs/t3_ssl_img_s0/emb_img.npy",
    "t3_spec": "runs/t3_ssl_joint_s0/emb_spec.npy",
    # epoch SimCLR untuk ketiga varian (citra, spektrum, joint)
    "t3_epochs": {"img": 100, "spec": 100, "joint": 100},
    "t3_k": 8,
    "t3_metode": "kmeans",
    "t3_w": 0.5,
}
PATH_FINAL = os.path.join(W, "final.json")


def py(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script)] + [str(a) for a in args]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})


def muat_final():
    cfg = dict(FINAL)
    if os.path.exists(PATH_FINAL):
        with open(PATH_FINAL, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


# ======================================================================== retrain
def retrain(task, data, cfg):
    ex = cfg["t1_extra"]
    if task == "1":
        py("task1_eda.py", "--data", data)
        py("task1_tekstur.py", "--data", data)       # analisis Fourier lini A vs B (dasar augmentasi tekstur)
        py("task1_eksperimen.py", "--data", data, "--grup", "sanity")
        py("task1_eksperimen.py", "--data", data, "--grup", "modality")
        py("task1_noise.py", "--data", data, "--fusion", cfg["t1_fusion"])  # OOF harus pakai fusion yang sama
        for g in ["noise", "domain", "missing"]:
            py("task1_eksperimen.py", "--data", data, "--grup", g, "--fusion", cfg["t1_fusion"])
        py("task1_eksperimen.py", "--data", data, "--grup", "final", "--fusion", cfg["t1_fusion"], "--extra", " ".join(ex))
        os.makedirs(os.path.join(W, "task1"), exist_ok=True)
        for s in range(3):
            shutil.copy(os.path.join(RUNS, f"t1_final_s{s}", "model.pt"), os.path.join(W, "task1", f"final_s{s}.pt"))
    elif task == "2":
        m, Wl, ep = cfg["t2_model"], cfg["t2_W"], cfg["t2_epochs"]
        py("task2_eda.py", "--data", data)
        py("task2_eksperimen.py", "--data", data, "--grup", "tune")      # dasar pemilihan resep DL
        py("task2_eksperimen.py", "--data", data, "--grup", "model", "--epochs", ep)
        py("task2_eksperimen.py", "--data", data, "--grup", "window", "--model", m, "--epochs", ep)
        py("task2_eksperimen.py", "--data", data, "--grup", "uncert", "--model", m, "--W", Wl, "--epochs", ep)
        py("task2_eksperimen.py", "--data", data, "--grup", "final", "--model", m, "--W", Wl, "--epochs", ep,
           "--n_seed", cfg["t2_seed"], "--geser", cfg["t2_geser"])
        # penamaan di task2_eksperimen.py: t2_cv_<m> = W 50 (default), t2_cv_<m>_W30 = W 30
        cv = f"t2_cv_{m}" + ("" if Wl == 50 else f"_W{Wl}")
        d = os.path.join(W, "task2")
        os.makedirs(d, exist_ok=True)
        for k in range(5):
            shutil.copy(os.path.join(RUNS, cv, f"model_fold{k}.pt"), os.path.join(d, f"cv_fold{k}.pt"))
            shutil.copy(os.path.join(RUNS, cv, f"prep_fold{k}.pkl"), os.path.join(d, f"cv_prep_fold{k}.pkl"))
        for s in range(cfg["t2_seed"]):
            shutil.copy(os.path.join(RUNS, f"t2_full_{m}_s{s}", "model.pt"), os.path.join(d, f"full_s{s}.pt"))
        shutil.copy(os.path.join(RUNS, f"t2_full_{m}_s0", "prep.pkl"), os.path.join(d, "full_prep.pkl"))
        shutil.copy(os.path.join(RUNS, f"t2_uncert_{m}", "metrics.json"), os.path.join(d, "uncert.json"))
    elif task == "3":
        py("task3_eda.py", "--data", data)
        for mod in ["img", "spec", "joint", "pretrained"]:
            ep = [] if mod == "pretrained" else ["--epochs", cfg["t3_epochs"][mod]]
            py("task3_ssl.py", "--data", data, "--modal", mod, "--nama", f"t3_ssl_{mod}_s0", *ep)
        py("task3_cluster.py", "--data", data, "--tahap", "banding", "--nama", "t3_cluster_banding")
        py("task3_cluster.py", "--data", data, "--tahap", "k", "--img", cfg["t3_img"], "--spec", cfg["t3_spec"],
           "--w", cfg["t3_w"], "--nama", "t3_cluster_k")
        py("task3_cluster.py", "--data", data, "--tahap", "final", "--k", cfg["t3_k"], "--metode", cfg["t3_metode"],
           "--img", cfg["t3_img"], "--spec", cfg["t3_spec"], "--w", cfg["t3_w"], "--nama", "t3_cluster_final")
        salin_encoder_task3(cfg)


def run_dari_path(p):
    """'runs/t3_ssl_joint_s0/emb_img.npy' -> 't3_ssl_joint_s0'."""
    return os.path.basename(os.path.dirname(p.replace("\\", "/")))


def salin_encoder_task3(cfg):
    """Encoder citra & spektrum bisa berasal dari run berbeda (pilihan di t3_cluster_banding).
    weights/task3/{img,spec}/ berisi encoder + norm.json run asalnya; sumber.json mencatat asalnya.
    Citra dari ResNet18 pretrained tidak perlu disalin (bobot torchvision dimuat ulang)."""
    d = os.path.join(W, "task3")
    sumber = {}
    for mod, f in [("img", "enc_img.pt"), ("spec", "enc_spec.pt")]:
        run = run_dari_path(cfg[f"t3_{mod}"])
        sumber[mod] = run
        os.makedirs(os.path.join(d, mod), exist_ok=True)
        if "pretrained" in run:
            continue
        for x in [f, "norm.json"]:
            shutil.copy(os.path.join(RUNS, run, x), os.path.join(d, mod, x))
    simpan_json(sumber, os.path.join(d, "sumber.json"))


# ======================================================================== cepat
def cepat_task1(data, cfg):
    import torch

    from metrik import KELAS, f1_per_kelas
    from task1_data import LINI, bobot_lini_test, muat, split_validasi, tensor_gpu
    from task1_model import Task1Net
    from task1_train import prediksi

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, dte = muat(data, "train"), muat(data, "test")
    _, idx_va = split_validasi(d["meta"], 0.2, seed=0)
    w_lini = bobot_lini_test(d["meta"], dte["meta"])
    yv, lini = d["y"][idx_va], d["lini"][idx_va]
    rows, p_test = [], []
    for s in range(3):
        ck = torch.load(os.path.join(W, "task1", f"final_s{s}.pt"), map_location=dev, weights_only=False)
        a = ck["args"]
        model = Task1Net(a["modal"], a["fusion"]).to(dev)
        model.load_state_dict(ck["state_dict"])
        t = tensor_gpu(d, ck["stats"], a["norm"], dev)
        va = torch.from_numpy(idx_va).to(dev)
        pv = prediksi(model, t, va).argmax(1)
        pns = prediksi(model, t, va, hapus_sensor=True).argmax(1)
        r = {"seed": s, "macro_f1": float(np.nanmean(f1_per_kelas(yv, pv)))}
        for i, l in enumerate(LINI):
            r[f"macro_f1_lini_{l}"] = float(np.nanmean(f1_per_kelas(yv[lini == i], pv[lini == i])))
        r["macro_f1_bobot_test"] = float(np.nanmean(f1_per_kelas(yv, pv, bobot=w_lini[lini])))
        r["macro_f1_tanpa_sensor"] = float(np.nanmean(f1_per_kelas(yv, pns)))
        rows.append(r)
        tt = tensor_gpu(dte, ck["stats"], a["norm"], dev)
        p_test.append(prediksi(model, tt, torch.arange(len(dte["meta"]), device=dev)))
    df = pd.DataFrame(rows)
    print("\n[Task 1] validasi model final per seed:\n" + df.round(4).to_string(index=False))
    ms = df.drop(columns="seed").agg(["mean", "std"]).T
    print("mean ± std:\n" + ms.round(4).to_string())
    out = pd.DataFrame({"id_unit": dte["meta"]["id_unit"], "kelas": np.array(KELAS)[np.mean(p_test, 0).argmax(1)]})
    out.to_csv(os.path.join(ROOT, "prediksi_klasifikasi.csv"), index=False)
    print("prediksi_klasifikasi.csv ditulis:", out.kelas.value_counts().to_dict())
    return {"per_seed": rows, "mean": ms["mean"].to_dict(), "std": ms["std"].to_dict()}


def cepat_task2(data, cfg):
    import torch
    from sklearn.model_selection import GroupKFold

    from metrik import RUL_CAP, ringkasan_rul
    from task2_model import buat_model
    from task2_prep import Jendela, label_rul, muat, titik_potong
    from task2_train import prediksi_dl, prediksi_uji_final

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Wl = cfg["t2_W"]
    df = muat(data, "train")
    y = label_rul(df)
    potong = titik_potong(df, seed=0)
    oof = np.full(len(df), np.nan)
    for k, (itr, iva) in enumerate(GroupKFold(5).split(df, groups=df["id_unit"])):
        with open(os.path.join(W, "task2", f"cv_prep_fold{k}.pkl"), "rb") as f:
            prep = pickle.load(f)
        dva = df.iloc[iva].reset_index(drop=True)
        J = Jendela(prep.transform(dva), dva["id_unit"].values, Wl).ke_torch(dev)
        ck = torch.load(os.path.join(W, "task2", f"cv_fold{k}.pt"), map_location=dev, weights_only=False)
        model = buat_model(cfg["t2_model"], J.X.shape[1], Wl, ck["args"].get("dropout", 0.2)).to(dev)
        model.load_state_dict(ck["state_dict"])
        oof[iva] = prediksi_dl(model, J, np.arange(len(dva)), dev)[0]
    sel = np.zeros(len(df), bool)
    sel[potong] = True
    r = ringkasan_rul(y[sel], np.clip(oof[sel], 0, RUL_CAP))
    print(f"\n[Task 2] CV 5 fold ({cfg['t2_model']}, W={Wl}) titik potong: RMSE {r['rmse']:.3f} | skor {r['skor']:.1f} "
          f"| skor/titik {r['skor_per_mesin']:.3f}")
    for rg in ["rul_0_50", "rul_50_100", "rul_100_150"]:
        print(f"   {rg}: n={r[rg]['n']} RMSE {r[rg]['rmse']:.3f} skor {r[rg]['skor']:.1f}")

    # fungsi yang SAMA dengan task2_eksperimen.buat_prediksi (seed MC dropout tetap) -> file identik
    ids, mus, sds = prediksi_uji_final(cfg["t2_model"], Wl,
                                       [os.path.join(W, "task2", f"full_s{s}.pt") for s in range(cfg["t2_seed"])],
                                       os.path.join(W, "task2", "full_prep.pkl"), data, dev)
    mu = mus.mean(0)
    sd = np.sqrt(np.maximum((sds ** 2 + mus ** 2).mean(0) - mu ** 2, 1e-6))
    q = json.load(open(os.path.join(W, "task2", "uncert.json"), encoding="utf-8"))["faktor_conformal_q"]
    rul = np.clip(mu + cfg["t2_geser"], 0, RUL_CAP)
    pd.DataFrame({"id_unit": ids, "rul": np.round(rul, 2)}).to_csv(
        os.path.join(ROOT, "prediksi_rul.csv"), index=False)
    lo = np.clip(mu - q * sd, 0, RUL_CAP)
    print(f"prediksi_rul.csv ditulis: {len(rul)} mesin, rata-rata {rul.mean():.2f}; "
          f"{(lo <= 30).mean():.1%} disarankan servis (batas bawah 90% <= 30)")
    return {"cv_titik_potong": r, "test_rul_rata2": float(rul.mean())}


def cepat_task3(data, cfg):
    import torch
    from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score

    from task3_cluster import GaussianMixture, PCA, dec, fusi, kmeans, ruang, sil
    from task3_ssl import EncGambar, EncSpektrum, embed_gambar, embed_pretrained, embed_spektrum

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(data, "clustering.npz"))
    meta = pd.read_csv(os.path.join(data, "clustering_meta.csv"))
    d3 = os.path.join(W, "task3")
    sumber = json.load(open(os.path.join(d3, "sumber.json"), encoding="utf-8"))
    if "pretrained" in sumber["img"]:
        ei = embed_pretrained(z["gambar"], dev)       # ResNet18 ImageNet (torchvision), lihat README
    else:
        nm = json.load(open(os.path.join(d3, "img", "norm.json"), encoding="utf-8"))
        G = ((torch.from_numpy(z["gambar"]).to(dev).float()[:, None] / 255) - nm["g_mu"]) / nm["g_sd"]
        eg = EncGambar().to(dev)
        eg.load_state_dict(torch.load(os.path.join(d3, "img", "enc_img.pt"), map_location=dev))
        ei = embed_gambar(eg, G)
    nm = json.load(open(os.path.join(d3, "spec", "norm.json"), encoding="utf-8"))
    S = (torch.from_numpy(z["spektrum"].astype(np.float32)).to(dev) - nm["s_mu"]) / nm["s_sd"]
    es = EncSpektrum().to(dev)
    es.load_state_dict(torch.load(os.path.join(d3, "spec", "enc_spec.pt"), map_location=dev))
    Xi, Xs = ruang(ei), ruang(embed_spektrum(es, S))
    print(f"[Task 3] encoder citra: {sumber['img']}, encoder spektrum: {sumber['spec']}")
    X = fusi(Xi, Xs, cfg["t3_w"])
    k, m = cfg["t3_k"], cfg["t3_metode"]
    if m == "kmeans":
        lab = kmeans(X, k)
    elif m == "gmm":
        lab = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit_predict(PCA(16, random_state=0).fit_transform(X))
    else:
        lab = dec(X, k, seed=0)[0]
    pd.DataFrame({"id_sampel": meta["id_sampel"], "cluster": lab.astype(int)}).to_csv(os.path.join(ROOT, "cluster.csv"), index=False)
    r = {"k": k, "metode": m, "silhouette": sil(X, lab), "davies_bouldin": float(davies_bouldin_score(X, lab)),
         "nmi_mikroskop": float(normalized_mutual_info_score(lab, meta["mikroskop"])),
         "ukuran": np.bincount(lab).tolist()}
    print(f"\n[Task 3] {m} k={k}: silhouette {r['silhouette']:.4f} | DB {r['davies_bouldin']:.3f} | "
          f"NMI mikroskop {r['nmi_mikroskop']:.4f} | ukuran {r['ukuran']}")
    print("cluster.csv ditulis")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl", help="folder hasil ekstrak data_uts_dl.zip")
    ap.add_argument("--task", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--retrain", action="store_true", help="latih ulang dari nol (lama)")
    a = ap.parse_args()
    print("versi library:", json.dumps(versi_library(), indent=1))
    if not os.path.isdir(a.data):
        sys.exit(f"folder data tidak ditemukan: {a.data}")
    cfg = muat_final()
    print("konfigurasi final:", cfg)
    tasks = ["1", "2", "3"] if a.task == "all" else [a.task]
    hasil = {}
    for t in tasks:
        t0 = time.time()
        if a.retrain:
            retrain(t, a.data, cfg)
        hasil[f"task{t}"] = {"1": cepat_task1, "2": cepat_task2, "3": cepat_task3}[t](a.data, cfg)
        print(f"[task{t}] selesai dalam {time.time() - t0:.1f} s", flush=True)
    simpan_json(hasil, os.path.join(RUNS, "reproduce", "hasil.json"))
    if a.retrain:
        py("tabel_laporan.py")
    from cek_submission import SPEK, cek, ids_dari_data

    # cek format + himpunan ID harus sama persis dengan ID test/sampel di data
    cek([os.path.join(ROOT, n) for n in SPEK if os.path.exists(os.path.join(ROOT, n))], ids_dari_data(a.data))


if __name__ == "__main__":
    main()
