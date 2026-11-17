# Kesimpulan

| Task | Model final | Hasil validasi utama (sumber) |
|---|---|---|
| 1 | Fusi cross-attention + modality dropout 25% + buang kandidat salah label, ensemble 3 seed | macro-F1 0,8700 ± 0,0012; tertimbang komposisi test 0,8286 ± 0,0015 (`runs/t1_tabel/final_mean_std.csv`) |
| 2 | GRU 2 lapis, W = 50, ensemble 5 seed + MC dropout, interval conformal | RMSE 19,54, skor asimetris 21,60 per titik (`runs/t2_cv_gru`); cakupan interval 90,3% (`runs/t2_uncert_gru`) |
| 3 | SimCLR (citra) + SimCLR joint (spektrum), fusi, K-Means k = 8 | silhouette 0,2293, stabilitas 3 seed 1,0000 (`runs/t3_cluster_final`) |

**Temuan utama.**
1. **Fusi multimodal diperlukan.** Pada Task 1, tiap modalitas hanya mengenali sebagian kelas (`las_dingin`
   hanya terlihat di sensor, `kontaminasi`/`noda`/`gores` hanya di gambar). Fusi menaikkan macro-F1 dari 0,687
   (gambar saja) ke 0,844. Pada Task 3, citra-saja dan spektrum-saja masing-masing menggabungkan pasangan
   material yang berbeda (ARI antara keduanya 0,4315). Hanya fusi yang memisahkan kedelapan cluster.
2. **Preprocessing dan validasi yang tepat lebih menentukan daripada arsitektur.** Pada Task 2, normalisasi
   per regime adalah syarat agar tren degradasi terlihat. Resep training yang ter-regularisasi menurunkan RMSE
   GRU dari 24,19 ke 19,54, sedangkan selisih antar arsitektur hanya 19,54–20,65. Pada Task 1, modality dropout
   menaikkan skor bahkan saat sensor lengkap (0,8440 → 0,8692).
3. **Ketidakpastian harus dikalibrasi.** Interval dari MC dropout dan ensemble hanya mencakup 62,2% (target
   90%). Split-conformal memperbaikinya menjadi 90,3%, dan interval itu menjadi dasar aturan servis
   "batas bawah 90% ≤ 30 siklus".

**Keterbatasan.**
- **Task 1**: celah antara lini A dan B (0,899 vs 0,780) belum teratasi. Tidak ada teknik domain yang kami uji
  yang menaikkan lini B, padahal lini B mencakup 60% data uji. Skor uji kemungkinan lebih dekat ke estimasi
  tertimbang (±0,83).
- **Task 2**: semua model lemah di rentang RUL 50–100 (RMSE ±29,5, cakupan interval 72,3%). Resep training
  dipilih di fold 0, sehingga estimasi CV sedikit optimis. Pergeseran prediksi untuk menurunkan skor asimetris
  sengaja tidak dipakai karena akan menaikkan RMSE.
- **Task 3**: bukti bahwa pasangan cluster 1 dan 6 adalah dua material berbeda paling lemah. Jika keduanya
  satu material, k yang tepat adalah 7.
- **Umum**: label validasi Task 1 sendiri berasal dari inspektur dan ikut mengandung noise. Sebagian
  eksperimen pembanding hanya dijalankan dengan satu seed, sehingga selisih kecil sebaiknya dibaca sebagai
  setara.
