# Persiapan Ujian Lisan

Format ujian: 5 menit reproduksi live (satu anggota dipilih dosen menjalankan model final dan menghasilkan ulang
satu angka laporan), lalu 15 menit pertanyaan tentang task **mana pun** kepada anggota **mana pun**. Semua angka
di bawah berasal dari `runs/*/metrics.json` (lihat bab laporan untuk sumber per angka).

## A. Reproduksi live (latih sebelum hari-H)

Persiapan, sekali saja:
```bash
git clone https://github.com/marcheliano-satrio/UTS_Deep_Learning.git && cd UTS_Deep_Learning
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r kode/requirements.txt
# ekstrak data_uts_dl.zip ke folder data_uts_dl/ di root repo
```

Saat ujian (mode cepat dari `weights/`, < 5 menit dengan GPU):
```bash
python kode/reproduce.py --data data_uts_dl --task 1     # macro-F1 3 seed: 0,8687 / 0,8711 / 0,8703 -> 0,8700 ± 0,0012
python kode/reproduce.py --data data_uts_dl --task 2     # CV 5 fold: RMSE 19,536 | skor 97.209,7 ; prediksi uji rata-rata 94,08
python kode/reproduce.py --data data_uts_dl --task 3     # K-Means k=8: silhouette 0,2293 | DB 1,782
python kode/cek_submission.py --data data_uts_dl         # cek format & himpunan ID ketiga file prediksi
```
Angka yang harus muncul sama dengan `runs/reproduce/hasil.json` dan tabel di laporan. Tanpa GPU, jalankan di
Colab (`kode/00_setup_colab.ipynb`). Latih ulang dari nol: tambahkan `--retrain` (beberapa jam, jangan saat ujian).

## B. Task 1 — Klasifikasi multimodal

1. **Kenapa macro-F1, dan bagaimana menangani kelas tidak seimbang?** Metrik di test tersembunyi adalah macro-F1,
   dan kelas sangat tidak seimbang (normal 49,8%, kontaminasi 4,9%). Loss diberi bobot kelas 1/√frekuensi agar
   kelas kecil tidak diabaikan.
2. **Strategi fusi apa saja, dan mana yang dipakai?** Concat (0,837), late (0,830), dan cross-attention
   (0,844). Cross-attention dipakai karena token gambar dan token sinyal bisa saling "bertanya" bagian mana yang
   relevan. Selisihnya kecil, jadi kesimpulan ini dibaca bersama variasi antar seed (std 0,0012).
3. **Kelas mana butuh modalitas mana?** `las_dingin` hanya terlihat di sensor arus (F1 gambar 0,000 vs sensor
   0,929). Kontaminasi, noda, dan gores hanya terlihat di gambar. Gelembung dan retak samar di keduanya, dan di
   kelas ini fusi memberi kenaikan terbesar.
4. **Bagaimana mendeteksi label salah?** Prediksi out-of-fold 5-fold, sehingga model tidak pernah melihat sampel
   yang dinilainya. Sampel ditandai bila p(label tercatat) < 0,05 sementara model > 0,80 yakin pada kelas lain:
   1.241 unit (2,6%). Pola utamanya normal ↔ las_dingin (tak terlihat visual) dan pasangan mirip visual.
5. **Teknik tahan noise mana yang berhasil?** Membuang 960 kandidat di bagian latih menaikkan macro-F1
   (0,8440 → 0,8521). Label smoothing tidak berpengaruh. Validasi tidak pernah diubah.
6. **Apa domain shift-nya?** Lini B hanya 25,1% data latih tetapi 60,0% data uji, dan F1 lini B jauh lebih
   rendah (0,754 vs 0,873). Analisis Fourier menunjukkan gambar lini B memuat garis halus berarah acak yang mirip
   goresan dan retak.
7. **Teknik domain apa yang diuji, dan kenapa tidak dipakai?** Normalisasi per lini, augmentasi fotometrik,
   bobot ulang, dan augmentasi garis tiruan. Tidak satu pun menaikkan lini B, jadi tidak dipakai. Ini hasil
   jujur, dan celah lini A–B menjadi keterbatasan utama.
8. **Bagaimana jika sensor tidak terekam?** Ada kanal mask, dan saat training digunakan modality dropout 25%
   (seluruh sensor dihapus). Tanpa sensor, macro-F1 0,671, mendekati model gambar saja (0,687), yang merupakan
   batas atas wajar. Bonusnya, dengan sensor lengkap skor juga naik (0,8440 → 0,8692).
9. **Apa gunanya sanity check overfit satu batch?** Untuk membuktikan pipeline data–model–loss benar: pada 64
   sampel, loss turun dari 1,96 ke 5,6·10⁻⁵ dan akurasi mencapai 1,000 setelah 300 langkah.
10. **Kenapa skor uji diperkirakan lebih rendah dari validasi?** Validasi 75% lini A, sedangkan test 60% lini B.
    Estimasi macro-F1 tertimbang komposisi test adalah 0,8286, bukan 0,8700.

## C. Task 2 — RUL (pemegang: M Marcheliano Taji Suryana)

1. **Kenapa normalisasi per regime, bukan global?** `setel_1` membentuk 6 regime operasi yang menggeser level
   sensor jauh lebih besar daripada degradasi (rasio std global/dalam-regime sampai 38,8). Normalisasi global
   mencampur kedua efek. Setelah normalisasi per regime, tren degradasi terlihat (korelasi s04–RUL −0,30 → −0,80).
   Statistiknya dihitung dari data latih saja, ulang di setiap fold (`Prep.fit`).
2. **Sensor mana yang dibuang, dan kenapa?** s01, s05, s06, s10, s16, s18, s19, karena |korelasi| dengan RUL
   ≤ 0,005 setelah normalisasi per regime. Tersisa 14 sensor, plus siklus/200 dan one-hot regime.
3. **Kenapa penanganan spike dan NaN harus kausal?** Data uji berhenti di tengah riwayat. Kalau spike dideteksi
   dengan median yang memakai siklus sesudahnya, saat latih kita "mengintip masa depan" yang tidak tersedia saat
   uji. Karena itu spike = selisih > 3 dari median 5 siklus **sebelumnya**, dan NaN diisi dari siklus
   sebelumnya (forward-fill).
4. **Bagaimana label dibuat dan kenapa cap 150?** RUL = min(150, siklus_terakhir − siklus). Di awal umur mesin
   belum terdegradasi, sehingga RUL 300 dan 200 tidak bisa dibedakan dari sensor. Cap 150 juga aturan soal.
5. **Bagaimana menangani riwayat yang lebih pendek dari window?** Padding nol di **kiri** ditambah kanal mask
   (0 = padding). Window hanya berisi siklus t−W+1…t, jadi tidak pernah ada siklus masa depan.
6. **Kenapa split per mesin?** Baris dari mesin yang sama dan siklus berdekatan hampir identik. Split per baris
   membuat model "menghafal" mesin validasi dan skor menjadi terlalu optimis. `GroupKFold` dengan grup
   `id_unit` memastikan mesin validasi tidak pernah terlihat saat training.
7. **Apa itu titik potong?** Data uji berhenti di titik acak sebelum rusak (riwayat ≥ 50 siklus). Untuk meniru
   hal itu, tiap mesin validasi dievaluasi di 5 siklus acak, bukan di semua siklus.
8. **Hasil perbandingan model?** GRU W 50 RMSE 19,54 dengan skor 97.210 (terbaik). Transformer 19,80 (skor
   119.409), TCN 20,65, LightGBM 20,13–20,82. Selisih GRU dan Transformer lebih kecil dari std antar fold, tetapi
   GRU unggul di skor asimetris dan 3,5× lebih cepat. Semua model lemah di RUL 50–100 (RMSE ±29,5).
9. **Apa itu skor asimetris, dan kenapa tidak menggeser prediksi?** d = prediksi − aktual. Bila d ≥ 0
   (terlambat/optimis) skornya e^(d/10) − 1, lebih berat daripada e^(−d/13) − 1 untuk d < 0, karena mesin bisa
   rusak sebelum diservis. Menggeser prediksi −15 siklus menurunkan skor 18.609 → 7.942 tetapi RMSE naik
   19,73 → 22,57. Kedua metrik dinilai, jadi kami tidak menggeser.
10. **Bagaimana interval prediksi dibuat, dan kapan mesin diservis?** Ensemble 5 seed × MC dropout 30. Interval
    mentah hanya mencakup 62,2%, karena hanya mengukur ketidakpastian model, bukan variasi antar mesin.
    Split-conformal memberi faktor q = 3,91 (bukan 1,645) dan cakupan menjadi 90,3%. Aturan servisnya: batas bawah
    interval 90% ≤ 30 siklus. Di validasi aturan ini menangkap 100% mesin yang akan rusak ≤ 10 siklus, dengan alarm
    palsu 1,9%. Prediksi titik ≤ 10 hanya menangkap 33%. Di data uji, 24% mesin disarankan servis sekarang.

Pertanyaan lanjutan yang mungkin muncul:
- *"Prediksi titik ≤ 30 juga menangkap 100% dengan servis lebih sedikit; kenapa memakai batas bawah?"* Sampel
  mesin hampir rusak di fold 0 hanya 39 titik, prediksi titik bias terlambat di rentang 50–100, dan cakupan
  interval di rentang itu baru 72,3%. Kerusakan mendadak lebih mahal daripada servis 20–25 siklus lebih awal,
  jadi kami memilih aturan yang lebih konservatif.
- *"Apakah ada bias dalam estimasi CV?"* Ada sedikit. Resep training dipilih di fold 0 dan fold 0 ikut dirata-rata.
  Analisis ketidakpastian juga di fold 0. Data uji tidak pernah dipakai untuk memilih apa pun.

## D. Task 3 — Clustering multimodal

1. **Distorsi apa yang ditemukan?** Pada citra: rotasi (orientasi tekstur merata), perbedaan mikroskop (M2 lebih
   terang dan tajam), skala, dan kecerahan. Pada spektrum: baseline miring/lengkung, spike (22,5% spektrum),
   dan pergeseran kecil.
2. **Kenapa SimCLR, dan bagaimana memilih augmentasinya?** Tidak ada label. SimCLR memaksa dua augmentasi dari
   sampel yang sama berdekatan, sehingga embedding mengabaikan apa pun yang diubah augmentasi. Augmentasi dipilih
   untuk meniru distorsi pengganggu (rotasi, kecerahan/blur mikroskop, baseline, spike), bukan perbedaan
   material. Geser spektrum dibuat kecil (±6) karena posisi puncak mungkin ciri material.
3. **Bagaimana memilih representasi tanpa label?** Dengan kesepakatan lintas modalitas: K-Means terpisah di
   embedding citra dan spektrum, lalu ARI antara keduanya. Material memengaruhi kedua modalitas, sedangkan
   pengganggu hanya salah satu.
4. **Kenapa tidak memakai ResNet ImageNet saja?** ARI-nya lebih rendah (0,4700 vs 0,5955 pada k = 6). Bobot
   ImageNet tidak mengenal tekstur mikrostruktur dan tidak invarian rotasi.
5. **Bagaimana kedua modalitas digabung?** Standardisasi → PCA 32 → normalisasi L2 → concat dengan bobot
   setara (w = 0,5).
6. **Apa yang hilang jika hanya satu modalitas?** Citra saja menggabungkan pasangan 2/7 dan 1/6 (tekstur mirip,
   beda spektrum). Spektrum saja menggabungkan 0/4 dan 3/5 (spektrum sama, beda tekstur). ARI citra-saja vs
   spektrum-saja hanya 0,4315.
7. **Metode clustering apa yang dibandingkan?** K-Means, GMM, dan DEC (deep clustering, varian IDEC). Silhouette
   K-Means dan DEC seri (0,2293 vs 0,2292), sedangkan stabilitas antar seed K-Means 1,0000 vs DEC 0,8835, jadi
   K-Means dipilih. Model deep learning-nya tetap encoder SimCLR.
8. **Kenapa k = 8, padahal silhouette maksimum di k = 5?** Strukturnya bertingkat. Silhouette dan DB 64-d
   menunjukkan 5 kelompok kasar, silhouette UMAP dan komponen graf kNN menunjukkan 7, dan uji belah per
   modalitas menemukan 3 kelompok kasar yang terbelah di satu modalitas. Jadi 5 + 3 = 8. K-Means dan DEC pada
   k = 8 hampir identik (ARI 0,9910).
9. **Cluster mana yang sulit dipisah?** 1 vs 6: citra sama-sama garis halus dan spektrum hanya berbeda tinggi
   relatif dua puncak tengah. Uji belahnya paling lemah (lembah 0,202). Jika keduanya satu material, k yang
   tepat adalah 7.
10. **Apakah cluster mengikuti mikroskop?** Tidak. NMI cluster–mikroskop ≈ 0, dan proporsi M2 per cluster
    0,388–0,407. Atribut mikroskop tidak pernah dipakai untuk training, hanya untuk memeriksa.

## E. Pertanyaan umum
- **Bagaimana kalian memakai AI dan memverifikasinya?** Lihat `kontribusi.md`: Claude Code membantu draf kode
  dan laporan. Verifikasi dilakukan dengan uji unit metrik, sanity check, smoke test, cek kebocoran dengan
  membaca kode, penelusuran setiap angka ke `metrics.json`, dan reproduksi ulang.
- **Dari mana sebuah angka di laporan?** Tunjukkan `runs/<run>/metrics.json` yang disebut di teks, atau
  `docs/tabel_laporan.md` (dibuat ulang dengan `python kode/tabel_laporan.py`).
- **Seed dan versi library?** Tersimpan di `config.json` setiap run. Training memakai Python 3.12.10,
  PyTorch 2.14.1 (CUDA 12.6), dan RTX 3050 Laptop.
