"""Task 2a-c — preprocessing, label RUL, dan sliding window untuk data mesin press.

Temuan EDA (task2_eda.py) yang mendasari pilihan:
1. Regime operasi: setel_1 hanya bernilai di sekitar {0, 10, 20, 25, 35, 42} (setel_2 dan
   setel_3 ikut ditentukan olehnya) -> 6 regime. Regime menggeser level sensor jauh lebih
   besar daripada degradasi (std global s/d 10x std di dalam regime).
   => NORMALISASI PER REGIME: z = (x - mean_regime) / std_regime, mean/std dihitung dari
   data latih saja (dari fold latih saat validasi silang). Setelah itu sinyal degradasi
   terlihat sebagai tren halus, bukan lompatan antar regime. Min-max global tidak dipakai
   karena akan mencampur efek regime dan degradasi.
2. Sensor tidak informatif: setelah normalisasi per regime, s01, s05, s06, s10, s16, s18,
   s19 hampir tidak berkorelasi dengan RUL (|r| < 0.01) -> dibuang. Sisa 14 sensor.
3. Nilai hilang ~2% per sensor secara acak -> forward-fill di dalam mesin yang sama
   (kausal: hanya memakai nilai masa lalu), sisa NaN di awal riwayat diisi 0 (= rata-rata
   regime).
4. Spike: titik tunggal yang menyimpang > 3 (satuan z) dari median 5 siklus SEBELUMNYA
   diganti median itu. Hanya memakai masa lalu, sehingga identik saat inferensi pada data
   uji yang terpotong (tidak ada kebocoran masa depan).

Fitur per siklus: 14 sensor ternormalisasi + siklus/200 + one-hot regime (6) = 21.
Label: RUL = min(150, siklus_terakhir - siklus) (siklus terakhir di data latih = rusak).
Window: panjang W yang berakhir di setiap siklus t memakai siklus t-W+1..t. Riwayat lebih
pendek dari W di-padding nol di KIRI dengan kanal mask = 0 (bukan mengintip masa depan).
"""
import os

import numpy as np
import pandas as pd

SENSOR = [f"s{i:02d}" for i in range(1, 22)]
SENSOR_BUANG = ["s01", "s05", "s06", "s10", "s16", "s18", "s19"]
SENSOR_PAKAI = [s for s in SENSOR if s not in SENSOR_BUANG]
PUSAT_REGIME = np.array([0, 10, 20, 25, 35, 42], float)
RUL_CAP = 150
AMBANG_SPIKE = 3.0


def muat(data_dir, split):
    return pd.read_csv(os.path.join(data_dir, f"rul_{split}.csv.gz")).sort_values(["id_unit", "siklus"]).reset_index(drop=True)


def regime(df):
    """Indeks regime 0..5 = pusat setel_1 terdekat."""
    return np.abs(df["setel_1"].values[:, None] - PUSAT_REGIME[None]).argmin(1)


def label_rul(df):
    akhir = df.groupby("id_unit")["siklus"].transform("max")
    return np.minimum(RUL_CAP, akhir - df["siklus"]).values.astype(np.float32)


class Prep:
    """fit() pada data latih saja; transform() pada data apa pun (latih/validasi/uji)."""

    def __init__(self, sensor=SENSOR_PAKAI, ambang_spike=AMBANG_SPIKE):
        self.sensor = list(sensor)
        self.ambang = ambang_spike

    def fit(self, df):
        r = regime(df)
        x = df[self.sensor].values
        self.mu = np.stack([np.nanmean(x[r == k], 0) for k in range(6)])
        self.sd = np.stack([np.nanstd(x[r == k], 0) for k in range(6)])
        return self

    def transform(self, df, kembalikan_info=False):
        r = regime(df)
        z = (df[self.sensor].values - self.mu[r]) / self.sd[r]
        z = pd.DataFrame(z, columns=self.sensor)
        unit = df["id_unit"].values
        # spike kausal: median 5 nilai sebelumnya (dalam mesin yang sama)
        prev_med = z.groupby(unit).transform(lambda v: v.shift(1).rolling(5, min_periods=3).median())
        spike = (z - prev_med).abs() > self.ambang
        n_spike = int(spike.values.sum())
        z = z.mask(spike, prev_med)
        n_nan = int(z.isna().values.sum())
        z = z.groupby(unit).ffill().fillna(0.0)
        f = np.concatenate([
            z.values,
            (df["siklus"].values / 200.0)[:, None],
            np.eye(6)[r],
        ], 1).astype(np.float32)
        if kembalikan_info:
            return f, {"n_spike_diganti": n_spike, "n_nan_diisi": n_nan}
        return f

    @property
    def n_fitur(self):
        return len(self.sensor) + 1 + 6


class Jendela:
    """Penyimpanan window efisien: tiap mesin diberi W-1 baris padding di kiri lalu window
    untuk baris ke-r adalah potongan [awal_r, awal_r + W). Kanal terakhir = mask (1 = data asli).
    """

    def __init__(self, fitur, unit, W):
        self.W = W
        blok, awal, pos = [], np.empty(len(unit), np.int64), 0
        unit = np.asarray(unit)
        batas = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1], True])
        F = fitur.shape[1]
        for a, b in zip(batas[:-1], batas[1:]):
            pad = np.zeros((W - 1, F + 1), np.float32)
            asli = np.concatenate([fitur[a:b], np.ones((b - a, 1), np.float32)], 1)
            blok += [pad, asli]
            # window yang berakhir di baris i (indeks global a+j) dimulai di pos + j
            awal[a:b] = pos + np.arange(b - a)
            pos += W - 1 + (b - a)
        self.X = np.concatenate(blok, 0)
        self.awal = awal

    def ke_torch(self, device):
        import torch

        self.Xt = torch.from_numpy(self.X).to(device)
        self.awal_t = torch.from_numpy(self.awal).to(device)
        self.ar = torch.arange(self.W, device=device)
        return self

    def batch(self, idx):
        """idx: tensor indeks baris -> (B, W, F+1)."""
        return self.Xt[self.awal_t[idx][:, None] + self.ar[None]]

    def numpy_batch(self, idx):
        return self.X[self.awal[idx][:, None] + np.arange(self.W)[None]]


def titik_potong(df, n_per_mesin=5, min_siklus=50, seed=0):
    """Indeks baris 'titik potong' validasi yang meniru data uji: data uji berhenti di titik
    acak sebelum rusak dan panjang riwayat uji minimal 50 siklus. Untuk tiap mesin validasi
    diambil n titik acak seragam di [min_siklus, siklus_terakhir]."""
    rng = np.random.RandomState(seed)
    out = []
    for _, g in df.groupby("id_unit"):
        kand = g.index[g["siklus"] >= min(min_siklus, g["siklus"].max())]
        out += list(rng.choice(kand, min(n_per_mesin, len(kand)), replace=False))
    return np.sort(np.array(out))


def baris_terakhir(df):
    return df.groupby("id_unit").tail(1).index.values
