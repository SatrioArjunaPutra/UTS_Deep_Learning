# Task 1 — klasifikasi cacat multimodal (gambar + sensor)

Model utama: CNN gambar + 1D-CNN sensor dengan fusion cross-attention (PyTorch, dilatih dari nol).
Konfigurasi final ada di `FINAL` pada `kode/reproduce.py`.

| File | Isi |
|---|---|
| `task1_eda.py` | 1a — eksplorasi data: contoh gambar, sinyal per kelas, sensor hilang, perbedaan lini A vs B. |
| `task1_data.py` | Pemuatan data, normalisasi (statistik dari train saja), split validasi stratified per (kelas, lini), batch di GPU. |
| `task1_model.py` | Encoder gambar, encoder sinyal, tiga strategi fusion (`concat`, `late`, `xattn`). |
| `task1_train.py` | Satu skrip training/evaluasi untuk semua eksperimen (`--modal`, `--fusion`, `--seed`, `--smoke`, `--overfit_batch`). |
| `task1_noise.py` | 1d — deteksi kandidat salah label dengan prediksi out-of-fold 5-fold. |
| `task1_tekstur.py` | 1e — analisis Fourier tekstur lini A vs B (dasar augmentasi tekstur). |
| `task1_eksperimen.py` | Menjalankan grup eksperimen `--grup {sanity,modality,noise,domain,missing,final,tabel}` dan membuat tabel/figur. |

Urutan: `task1_eda.py` → `task1_tekstur.py` → `task1_eksperimen.py --grup sanity` → `modality` →
`task1_noise.py` → `noise` → `domain` → `missing` → `final`.

```bash
python kode/task1_klasifikasi/task1_train.py --data data_kecil --smoke      # uji cepat CPU
python kode/reproduce.py --data data_uts_dl --task 1                         # angka laporan dari weights/
```

Output: `runs/t1_*/` (config.json, metrics.json, kurva), `laporan_figs/task1/`, `prediksi_klasifikasi.csv`.
