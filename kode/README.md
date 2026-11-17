# Kode UTS Deep Learning — Klasifikasi, Regresi, Clustering

## 1. Instalasi

Diuji dengan Python 3.12, Windows 11, GPU NVIDIA RTX 3050 Laptop (4 GB), CUDA 12.6.

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/Mac: source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r kode/requirements.txt
```

Versi library yang menghasilkan semua angka laporan: torch 2.14.1, torchvision 0.29.1, numpy 2.5.4,
pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, umap-learn 0.6.0, lightgbm 4.7.0
(juga tercatat otomatis di `runs/<nama>/config.json` tiap run).

Di Google Colab: buka `kode/00_setup_colab.ipynb` (clone repo, unduh data, jalankan).

## 2. Data

Unduh `data_uts_dl.zip` dari folder dosen
(https://drive.google.com/drive/folders/15TiPGfsw0Irp36QM_zwd_oZnfcQquQ9x), ekstrak sehingga
file-file berada langsung di `data_uts_dl/` (root repo):

```
data_uts_dl/
  kamus_data.md
  klasifikasi_train.npz  klasifikasi_train_meta.csv  klasifikasi_test.npz  klasifikasi_test_meta.csv
  rul_train.csv.gz  rul_test.csv.gz
  clustering.npz  clustering_meta.csv
```

Data tidak di-commit (lihat `.gitignore`). `python kode/buat_subset.py` membuat `data_kecil/`
untuk uji cepat di CPU.

## 3. Satu perintah reproduksi

```bash
python kode/reproduce.py --data data_uts_dl --task all             # cepat: dari weights/ (< 5 menit di GPU)
python kode/reproduce.py --data data_uts_dl --task all --retrain   # latih ulang SEMUA eksperimen (beberapa jam)
```

Mode cepat memuat weights final di `weights/`, menghitung ulang angka validasi model final,
menulis ketiga file prediksi di root repo (`prediksi_klasifikasi.csv`, `prediksi_rul.csv`,
`cluster.csv`), lalu menjalankan `cek_submission.py` (format + himpunan ID harus sama persis
dengan data). Ringkasan angka disimpan di `runs/reproduce/hasil.json`.

Satu task saja: `--task 1`, `--task 2`, atau `--task 3`. Cek file prediksi secara terpisah:
`python kode/cek_submission.py --data data_uts_dl`.

Waktu mode cepat `--task all` di laptop RTX 3050 4 GB: ± 70 detik total (Task 1 ± 14 s, Task 2 ± 4 s,
Task 3 ± 43 s, sisanya memuat data dan `cek_submission`).

## 4. Konfigurasi final, seed, dan determinisme

Konfigurasi final ada di `FINAL` di `kode/reproduce.py` (dipilih dari validasi, bukan dari data uji).

| Task | Model final | Seed |
|---|---|---|
| 1 | fusion gambar+sensor (`--fusion` di `FINAL`), ensemble 3 seed | 0, 1, 2 |
| 2 | sekuens (`t2_model` di `FINAL`), 5 seed × MC dropout 30 | 0–4 (CV fold k memakai seed 0 + 100k) |
| 3 | encoder citra SimCLR (`t3_ssl_img_s0`) + encoder spektrum SimCLR-joint (`t3_ssl_joint_s0`), fusi w = 0,5, K-Means k = 8 | 0 |

Setiap run menyimpan `runs/<nama>/config.json` (argumen + versi library) dan `metrics.json`
(sumber semua angka laporan; `docs/tabel_laporan.md` dibuat otomatis oleh `kode/tabel_laporan.py`).
Split validasi memakai seed tetap (Task 1: stratified per kelas+lini, seed 0; Task 2: GroupKFold
per mesin), jadi semua eksperimen dibandingkan di himpunan validasi yang sama.

Catatan: mode cepat (dari weights) memberi angka yang sama persis. `--retrain` di GPU tidak
100% deterministik (cuDNN), sehingga angka dapat berbeda sedikit di desimal ketiga/keempat.

## 5. Weights

Weights final disalin otomatis oleh `--retrain` ke `weights/` dan di-commit (total ± 38 MB,
tiap file < 100 MB):

```
weights/final.json                      # opsional: menimpa FINAL di reproduce.py
weights/task1/final_s{0,1,2}.pt         # state_dict + argumen + statistik normalisasi (dari train)
weights/task2/cv_fold{0..4}.pt, cv_prep_fold{0..4}.pkl, full_s{0..4}.pt, full_prep.pkl, uncert.json
weights/task3/sumber.json               # run asal encoder citra & spektrum (bisa berbeda)
weights/task3/img/enc_img.pt, norm.json  # tidak ada jika citra = ResNet18 pretrained (dimuat dari torchvision)
weights/task3/spec/enc_spec.pt, norm.json
```

## 6. Pretrained weights publik (wajib disitasi)

- Task 3, pembanding embedding gambar: ResNet18 bobot ImageNet dari torchvision
  (He dkk., 2016, *Deep Residual Learning for Image Recognition*, CVPR;
  https://download.pytorch.org/models/resnet18-f37072fd.pth), dipakai tanpa fine-tuning.
- Tidak ada data eksternal yang dipakai. Model lain dilatih dari nol.

## 7. Struktur skrip

| Task | Skrip (urutan) |
|---|---|
| umum | `common.py`, `metrik.py`, `cek_submission.py`, `buat_subset.py`, `ringkas_data.py`, `tabel_laporan.py` |
| 1 | `task1_eda.py` → `task1_data.py`/`task1_model.py`/`task1_train.py` → `task1_noise.py` → `task1_tekstur.py` (analisis Fourier lini A vs B) → `task1_eksperimen.py --grup {sanity,modality,noise,domain,missing,final}` |
| 2 | `task2_eda.py` → `task2_prep.py`/`task2_model.py`/`task2_train.py` → `task2_eksperimen.py --grup {tune,model,window,uncert,final}` |
| 3 | `task3_eda.py` → `task3_ssl.py --modal {img,spec,joint,pretrained}` → `task3_cluster.py --tahap {banding,k,final}` |

Uji cepat di CPU: `--data data_kecil --smoke` (subset dari `buat_subset.py`).
