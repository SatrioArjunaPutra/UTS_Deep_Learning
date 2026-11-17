# Pendahuluan

**Tujuan.** Laporan ini menjawab tiga task UTS Deep Learning: (1) klasifikasi cacat produk dari gambar dan
sinyal sensor, (2) prediksi sisa umur pakai (RUL) mesin press dari riwayat sensor per siklus, dan
(3) pengelompokan sampel material tanpa label dari citra mikrostruktur dan spektrum komposisi. Untuk setiap task,
kami membangun pipeline deep learning yang lengkap dan jujur: eksplorasi data, desain arsitektur, validasi yang
bebas kebocoran, analisis ketangguhan atau ketidakpastian, dan prediksi untuk data uji.

**Data** (`docs/kamus_data.md`, tanpa data eksternal):

| Task | Data latih | Data uji / evaluasi | Metrik di data tersembunyi |
|---|---|---|---|
| 1. Klasifikasi multimodal | 48.000 unit: gambar 64×64 + sinyal 3 kanal × 128, 7 kelas | 16.000 unit | macro-F1 |
| 2. RUL | 900 mesin, riwayat sampai rusak (254.174 siklus) | 300 mesin, terpotong sebelum rusak | RMSE + skor asimetris |
| 3. Clustering multimodal | 36.000 sampel: citra 64×64 + spektrum 128 titik, tanpa label | sampel yang sama | ARI |

**Ringkasan pendekatan.**
- **Task 1**: encoder CNN (gambar) dan 1D-CNN (sinyal) dengan tiga strategi fusi. *Cross-attention* adalah yang
  terbaik. Eksperimen pendukung: nilai tiap modalitas, deteksi label salah lewat prediksi out-of-fold, pengukuran
  domain shift per lini produksi, dan *modality dropout* agar model tetap masuk akal saat sensor hilang.
- **Task 2**: normalisasi per regime operasi, penanganan spike dan NaN yang kausal, label RUL dengan cap 150, dan
  sliding window kausal. GRU, TCN, dan Transformer dibandingkan dengan baseline LightGBM dalam validasi silang per
  mesin. Interval prediksi berasal dari ensemble dan MC dropout yang dikalibrasi dengan *split-conformal*, lalu
  dipakai untuk aturan servis.
- **Task 3**: representasi tanpa label dengan SimCLR. Augmentasinya dirancang dari distorsi yang ditemukan saat
  eksplorasi. Embedding citra dan spektrum difusi, lalu K-Means, GMM, dan DEC dibandingkan. Jumlah cluster
  ditentukan dengan beberapa kriteria.

**Reproduksibilitas dan konvensi angka.** Setiap angka di laporan berasal dari `runs/<run>/metrics.json` yang
dihasilkan kode. File sumbernya disebut di teks, dan tabel lengkapnya ada di `docs/tabel_laporan.md`, yang dibuat
otomatis oleh `kode/tabel_laporan.py`. Semua angka dapat dihitung ulang dengan
`python kode/reproduce.py --data data_uts_dl --task all`. Mode cepatnya memakai weights final di `weights/`,
sedangkan `--retrain` melatih ulang semuanya. Seed dan versi library disimpan di `config.json` setiap run.
Training dilakukan dengan Python 3.12.10, PyTorch 2.14.1 (CUDA 12.6), dan GPU RTX 3050 Laptop 4 GB
(`kode/requirements.txt`). Angka desimal ditulis dengan koma.
