"""Task 1 — pemuatan data, normalisasi, split validasi, dan batch di GPU.

Temuan EDA yang mendasari pilihan di sini (lihat task1_eda.py):
- 48.000 unit latih, 16.000 unit uji. Gambar uint8 64x64, sensor float16 (3, 128).
- Sensor hilang per KANAL secara utuh (tidak pernah sebagian): ~6% per kanal, ~5% unit
  kehilangan ketiganya. Maka masukan sensor = 3 kanal nilai + 3 kanal mask (1 = terekam),
  NaN diisi 0 setelah normalisasi (= rata-rata latih).
- Lini B hanya 25% data latih tetapi 60% data uji. Gambar lini B lebih gelap (rata-rata
  ~112 vs ~140), kontras lebih tinggi, dan punya garis halus berarah (periode ±4 px, arah acak per gambar;
  lihat analisis Fourier di task1_train.py); suhu (kanal 0) lini B ~2 C lebih tinggi.
  => opsi normalisasi per lini (statistik dari data latih per lini) dan bobot ulang lini.

Data seluruhnya kecil (~240 MB), jadi disimpan sebagai tensor di GPU dan di-batch langsung
(lebih cepat daripada DataLoader). Normalisasi gambar dilakukan per batch agar gambar tetap
uint8 di memori GPU.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrik import KELAS  # noqa: E402

LINI = ["A", "B"]
PROPORSI_LINI_TEST = None  # dihitung dari meta test saat dimuat (bukan label, aman dipakai)


def muat(data_dir, split="train"):
    """Kembalikan dict: gambar (N,64,64) uint8, sensor (N,3,128) float32 (NaN = hilang), meta."""
    z = np.load(os.path.join(data_dir, f"klasifikasi_{split}.npz"))
    meta = pd.read_csv(os.path.join(data_dir, f"klasifikasi_{split}_meta.csv"))
    assert (meta["indeks"].values == np.arange(len(meta))).all(), "indeks meta tidak urut"
    d = {"gambar": z["gambar"], "sensor": z["sensor"].astype(np.float32), "meta": meta}
    d["lini"] = meta["lini"].map({l: i for i, l in enumerate(LINI)}).values.astype(np.int64)
    if "kelas" in meta:
        d["y"] = meta["kelas"].map({k: i for i, k in enumerate(KELAS)}).values.astype(np.int64)
    return d


def hitung_statistik(d, idx):
    """Statistik normalisasi dari sampel latih `idx` saja (anti-kebocoran).

    gambar: mean/std piksel global dan per lini.
    sensor: mean/std per kanal (abaikan NaN), global dan per lini.
    """
    st = {}
    g = d["gambar"][idx].astype(np.float32) / 255.0
    s = d["sensor"][idx]
    lini = d["lini"][idx]
    st["img_global"] = (float(g.mean()), float(g.std()))
    st["sen_global"] = (np.nanmean(s, axis=(0, 2)), np.nanstd(s, axis=(0, 2)))
    for i, l in enumerate(LINI):
        m = lini == i
        st[f"img_{l}"] = (float(g[m].mean()), float(g[m].std()))
        st[f"sen_{l}"] = (np.nanmean(s[m], axis=(0, 2)), np.nanstd(s[m], axis=(0, 2)))
    return st


def tensor_gpu(d, st, norm="global", device="cuda"):
    """Siapkan tensor di device.

    norm='global'  : satu mean/std untuk semua lini.
    norm='perline' : mean/std dari lini masing-masing (teknik domain shift).
    Hasil: img uint8 (N,1,64,64), img_mu/img_sd per sampel, sen (N,3,128) ternormalisasi
    dengan NaN->0, mask (N,3) float.
    """
    import torch

    n = len(d["meta"])
    img_mu = np.empty(n, np.float32)
    img_sd = np.empty(n, np.float32)
    sen_mu = np.empty((n, 3), np.float32)
    sen_sd = np.empty((n, 3), np.float32)
    for i, l in enumerate(LINI):
        m = d["lini"] == i
        ki, ks = (f"img_{l}", f"sen_{l}") if norm == "perline" else ("img_global", "sen_global")
        img_mu[m], img_sd[m] = st[ki]
        sen_mu[m], sen_sd[m] = st[ks][0], st[ks][1]
    s = (d["sensor"] - sen_mu[:, :, None]) / sen_sd[:, :, None]
    mask = (~np.isnan(d["sensor"]).all(axis=2)).astype(np.float32)
    s = np.nan_to_num(s, nan=0.0).astype(np.float32)
    t = {
        "img": torch.from_numpy(d["gambar"][:, None]).to(device),
        "img_mu": torch.from_numpy(img_mu).to(device),
        "img_sd": torch.from_numpy(img_sd).to(device),
        "sen": torch.from_numpy(s).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "lini": torch.from_numpy(d["lini"]).to(device),
    }
    if "y" in d:
        t["y"] = torch.from_numpy(d["y"]).to(device)
    return t


def ambil_batch(t, idx):
    """Ambil batch dari tensor GPU, gambar dinormalisasi menjadi float."""
    img = (t["img"][idx].float() / 255.0 - t["img_mu"][idx, None, None, None]) / t["img_sd"][idx, None, None, None]
    b = {"img": img, "sen": t["sen"][idx], "mask": t["mask"][idx], "lini": t["lini"][idx],
         "img_sd": t["img_sd"][idx]}
    if "y" in t:
        b["y"] = t["y"][idx]
    return b


def split_validasi(meta, frac=0.2, seed=0):
    """Split stratified per (kelas, lini) supaya validasi punya kedua lini secara proporsional.

    Info lini disimpan di meta sehingga metrik per lini bisa dihitung.
    """
    from sklearn.model_selection import train_test_split

    strat = meta["kelas"].astype(str) + "_" + meta["lini"].astype(str)
    if strat.value_counts().min() < 2:      # hanya terjadi di subset kecil (data_kecil)
        strat = meta["kelas"].astype(str)
    idx = np.arange(len(meta))
    tr, va = train_test_split(idx, test_size=frac, random_state=seed, stratify=strat)
    return np.sort(tr), np.sort(va)


def bobot_lini_test(meta_train, meta_test):
    """Bobot per lini = proporsi di test / proporsi di latih (importance weighting).

    Dipakai untuk (a) metrik validasi tertimbang yang meniru komposisi test dan
    (b) opsi --line_reweight saat training.
    """
    p_tr = meta_train["lini"].value_counts(normalize=True)
    p_te = meta_test["lini"].value_counts(normalize=True)
    return np.array([p_te.get(l, 0) / p_tr.get(l, 1) for l in LINI], np.float32)
