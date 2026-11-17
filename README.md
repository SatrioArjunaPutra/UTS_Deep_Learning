# UTS Deep Learning 2026

Pipeline deep learning untuk tiga masalah pada data industri: klasifikasi cacat multimodal,
prediksi Remaining Useful Life (RUL) mesin, dan clustering material tanpa label.
Semua model utama dibangun dengan PyTorch. Metode klasik hanya dipakai sebagai baseline.

**Kelompok:** M Marcheliano Taji Suryana dan Satrio Arjuna Putra

| Anggota | Bagian |
|---|---|
| Satrio Arjuna Putra | Task 1, Task 3, integrasi (`kode/reproduce.py`, weights, file prediksi) |
| M Marcheliano Taji Suryana | Task 2, laporan (`laporan.pdf`), `kontribusi.md` |

## Ringkasan task

| Task | Masalah | Data | Model utama (deep learning) | File prediksi |
|---|---|---|---|---|
| 1 | Klasifikasi 7 kelas cacat | Gambar 64×64 + sinyal sensor 3 kanal | CNN gambar + 1D-CNN sensor, fusion cross-attention | `prediksi_klasifikasi.csv` |
| 2 | Regresi RUL mesin press | Deret waktu sensor per siklus | Model sekuens (GRU) dengan window kausal, MC dropout untuk ketidakpastian | `prediksi_rul.csv` |
| 3 | Clustering material tanpa label | Gambar mikrostruktur + spektrum komposisi | Encoder SimCLR gambar & spektrum, fusi embedding, K-Means | `cluster.csv` |

**Penjelasan tiap task** (masalah, alur model, alasan setiap pilihan, hasil, cara menjalankan):

- Task 1 → [`kode/task1_klasifikasi/README.md`](kode/task1_klasifikasi/README.md)
- Task 2 → [`kode/task2_rul/README.md`](kode/task2_rul/README.md)
- Task 3 → [`kode/task3_cluster/README.md`](kode/task3_cluster/README.md)

Hasil, analisis, dan semua angka ada di `laporan.pdf`. Setiap angka berasal dari
`runs/<nama_run>/metrics.json` yang dihasilkan kode.

## Struktur repo

```
.
├── laporan.pdf                  # laporan Task 1–3 (maks. 18 halaman)
├── kontribusi.md                # peran anggota, link PR, penggunaan AI
├── prediksi_klasifikasi.csv     # id_unit,kelas
├── prediksi_rul.csv             # id_unit,rul
├── cluster.csv                  # id_sampel,cluster
├── kode/                        # seluruh kode + README (lihat kode/README.md)
│   ├── reproduce.py             # satu perintah untuk semua angka laporan
│   ├── common/                  # utilitas bersama
│   ├── task1_klasifikasi/
│   ├── task2_rul/
│   └── task3_cluster/
├── weights/                     # weights model final (< 100 MB per file)
├── runs/                        # config.json + metrics.json tiap eksperimen
├── laporan/, laporan_figs/      # bab laporan (markdown) dan figur
└── docs/                        # soal, kamus data, catatan kerja
```

## Cara menjalankan

**Paling mudah: Google Colab.** Klik tombol di bawah, pilih Runtime → Change runtime type → T4 GPU,
lalu jalankan sel berurutan. Notebook mengambil kode repo ini, mengunduh data dari tautan dosen,
dan menjalankan `reproduce.py`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcheliano-satrio/UTS_Deep_Learning/blob/main/kode/00_setup_colab.ipynb)

**Di laptop/PC sendiri:**

1. Instal (Python 3.12):

   ```bash
   pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
   pip install -r kode/requirements.txt
   ```

2. Unduh `data_uts_dl.zip` dari dosen dan ekstrak ke `data_uts_dl/` di root repo.
   Data tidak disertakan di repo.

3. Reproduksi semua angka laporan dan ketiga file prediksi:

   ```bash
   python kode/reproduce.py --data data_uts_dl --task all             # cepat, dari weights/ (< 5 menit di GPU)
   python kode/reproduce.py --data data_uts_dl --task all --retrain   # latih ulang semua eksperimen
   ```

   Satu task saja: `--task 1`, `--task 2`, atau `--task 3`.

Detail instalasi, seed, versi library, weights, dan urutan skrip ada di [`kode/README.md`](kode/README.md).

## Catatan

- Tidak ada data eksternal. Satu-satunya pretrained weights publik (ResNet18 ImageNet, torchvision)
  hanya dipakai sebagai pembanding di Task 3 dan disitasi di `kode/README.md`.
- Normalisasi dihitung dari data latih saja. Split validasi Task 2 per mesin (GroupKFold).
  Data uji hanya dipakai untuk prediksi akhir.
