# Task 2 — Prediksi Remaining Useful Life (RUL) Mesin Press

Penulis bab: M Marcheliano Taji Suryana. Semua angka berasal dari `runs/<run>/metrics.json` (tabel model:
`runs/t2_tabel/model.csv`; tabel lengkap: `docs/tabel_laporan.md`). Kode di `kode/task2_rul/`.
Evaluasi utama memakai **titik potong**: untuk tiap mesin validasi diambil 5 siklus acak (siklus ≥ 50),
dan RUL diprediksi dari window yang berakhir di siklus itu. Cara ini meniru data uji, yang berhenti di
titik acak sebelum rusak dengan riwayat minimal 50 siklus (`task2_prep.titik_potong`).

## 2a. Eksplorasi dan preprocessing (`task2_eda.py`, `task2_prep.py`)

Data latih berisi 900 mesin (254.174 baris) dan data uji 300 mesin (51.692 baris). Panjang riwayat latih
rata-rata 282,4 siklus (min 133, median 265, maks 778). Riwayat uji lebih pendek, rata-rata 172,3
(min 50, median 162,5, maks 455). Nomor siklus berurutan tanpa lubang
(`runs/t2_eda/metrics.json`, Gambar `laporan_figs/task2/eda_panjang.png`, `eda_contoh_mesin.png`).

- **Setelan operasi → 6 regime** (`eda_regime.png`). `setel_1` hanya bernilai di sekitar
  {0, 10, 20, 25, 35, 42}, dan `setel_2`/`setel_3` ikut ditentukan olehnya. Simpangan baku setelan di dalam
  satu regime hanya 0,002. Jumlah baris per regime merata (42.078–42.640). Regime menggeser level sensor jauh
  lebih besar daripada degradasi: rasio std global terhadap std di dalam regime mencapai 38,8 untuk s21 dan
  21,1 untuk s19 (`runs/t2_eda/sensor.csv`).
- **Normalisasi per regime**: z = (x − μ_regime) / σ_regime, dengan μ dan σ dihitung dari **data latih saja**.
  Saat validasi silang, keduanya dihitung ulang dari fold latih (`Prep.fit`). Setelah normalisasi, tren
  degradasi baru tampak. Korelasi s04 dengan RUL naik dari −0,30 (mentah) ke −0,80, s21 dari 0,006 ke 0,34,
  dan s20 dari 0,04 ke 0,39 (`sensor.csv`). Min-max global tidak dipakai karena akan mencampur efek regime
  dengan efek degradasi. Indeks regime juga dimasukkan sebagai fitur one-hot (6 kanal).
- **Sensor tak informatif**: s01, s05, s06, s10, s16, s18, dan s19 hampir tidak berkorelasi dengan RUL
  setelah normalisasi per regime (|r| ≤ 0,005, `sensor.csv`), jadi dibuang. Tersisa 14 sensor.
- **Nilai hilang**: rata-rata ±2,0% per sensor, tersebar acak. Akibatnya 35,0% baris punya minimal satu NaN,
  sehingga membuang baris bukan pilihan. NaN diisi dengan *forward-fill* di dalam mesin yang sama (hanya nilai
  masa lalu). Sisa NaN di awal riwayat diisi 0, yaitu rata-rata regime.
- **Spike**: sebuah nilai dianggap spike bila menyimpang > 3 (satuan z) dari median 5 siklus
  **sebelumnya**, lalu diganti dengan median itu. Aturannya kausal, sehingga perlakuannya identik pada data
  uji yang terpotong dan tidak memakai masa depan. Di data latih penuh ada 5.199 spike yang diganti dan
  72.200 NaN yang diisi (`runs/t2_eda/metrics.json` → `preprocessing_train`).

Fitur per siklus = 14 sensor + siklus/200 + one-hot regime (6) = 21 kanal, ditambah 1 kanal mask.

## 2b. Label (`task2_prep.label_rul`)

Di data latih, siklus terakhir tiap mesin adalah siklus saat rusak, sehingga
RUL(t) = min(150, siklus_terakhir − t). Cap 150 mengikuti aturan soal dan juga masuk akal secara fisik: di
awal umur, mesin belum menunjukkan degradasi, sehingga RUL 300 dan RUL 200 tidak dapat dibedakan dari sensor.
Model memprediksi RUL/150. Saat evaluasi dan prediksi, keluaran di-*clip* ke [0, 150].

## 2c. Sekuens (`task2_prep.Jendela`)

Window panjang W yang berakhir di siklus t berisi siklus t−W+1 … t, jadi **tidak pernah memuat siklus
sesudah t**. Riwayat yang lebih pendek dari W diberi padding nol di **kiri**, dan kanal mask bernilai 0 di
bagian padding (1 = data asli). Model bisa membedakan padding dari bacaan asli tanpa perlu mengintip masa
depan. Satu window dibuat untuk setiap siklus setiap mesin latih. Window disimpan sekali per mesin dan
diambil lewat indeks, bukan disalin per sampel, sehingga hemat memori. Untuk data uji, prediksi dibuat dari
window yang berakhir di siklus terakhir tiap mesin (`baris_terakhir`).

## 2d. Model (`task2_model.py`, `task2_train.py`)

| Model | Arsitektur |
|---|---|
| **GRU** | GRU 2 lapis (hidden 64) → status di siklus terakhir → MLP (64→32→1). Kausal secara alami. |
| **TCN** | Konvolusi 1D kausal ber-dilasi (1, 2, 4, 8), kernel 3, residual, 64 kanal. Receptive field 61 ≥ W. |
| **Transformer** | Encoder 2 lapis (d = 64, 4 head, pre-norm), positional embedding dipelajari, padding di-mask. Token siklus terakhir sebagai ringkasan. |
| LightGBM (baseline non-DL) | Ringkasan window per sensor: nilai terakhir, rata-rata, std, kemiringan (hanya siklus asli), + siklus, regime, panjang riwayat. 600 pohon. |

Semua model DL memakai dropout (syarat MC dropout di 2f), loss MSE, AdamW + OneCycle, clip gradien 1,0,
dan batch 512. Model yang dievaluasi adalah model **epoch terakhir**, tanpa pemilihan epoch di validasi.

**Tuning resep** dilakukan di fold 0 saja (`runs/t2_tune_*`, RMSE titik potong). Resep GRU awal (W 30,
25 epoch, lr 2·10⁻³, tanpa noise) memberi RMSE validasi 24,19 di CV 5 fold (`runs/t2_cv_gru_awal`),
padahal train RMSE di epoch terakhir hanya 9,8–10,4 (`history.json`). Resep ini jelas overfit. Resep A (W 50, 15 epoch, dropout 0,3, wd 10⁻³, noise input 0,3) memberi
22,10. Resep C (25 epoch, noise 0,5) memberi 22,09. W 80 (resep A) justru memburuk menjadi 23,20. Resep **E**
(W 50, **10 epoch, lr 10⁻³**, dropout 0,3, wd 10⁻³, noise Gaussian 0,3 pada kanal sensor asli) memberi
**19,89** dan dipakai **sama untuk ketiga arsitektur** agar perbandingan adil. Pada resep A/B, TCN mendapat
21,00/22,02 dan Transformer 22,09/23,16. Dengan resep final, train RMSE GRU di epoch terakhir 18,6–20,1
(fold 0–1, `runs/t2_cv_gru/history.json`; noise augmentasi ikut menaikkannya), dekat dengan RMSE validasi,
sehingga tidak ada celah overfitting yang besar. Jadi pelajaran utamanya: pada data ini, regularisasi kuat dan
training singkat lebih penting daripada kapasitas model.

## 2e. Validasi per mesin (GroupKFold 5 fold)

Split memakai `GroupKFold` dengan grup = `id_unit`. Semua siklus satu mesin berada di fold yang sama,
sehingga model tidak pernah melihat mesin validasi. Split per baris akan bocor karena baris-baris yang
berdekatan dari satu mesin hampir identik. Skor asimetris dihitung dengan d = prediksi − aktual:
e^(−d/13) − 1 bila d < 0 (terlalu cepat) dan e^(d/10) − 1 bila d ≥ 0 (terlambat). Prediksi yang terlalu optimis
dihukum lebih berat karena mesin bisa rusak sebelum diservis. Total ada 4.500 titik potong (900 mesin × 5):
1.076 dengan RUL aktual 0–50, 1.052 dengan 50–100, dan 2.372 dengan 100–150.

| Model (`runs/t2_tabel/model.csv`) | RMSE | RMSE per fold (mean ± std) | Skor total | Skor / titik | RMSE 0–50 | RMSE 50–100 | RMSE 100–150 |
|---|---|---|---|---|---|---|---|
| **GRU, W 50** | **19,54** | 19,53 ± 0,71 | **97.210** | **21,60** | 11,95 | 29,50 | 16,53 |
| GRU, W 30 | 19,89 | 19,88 ± 0,75 | 103.599 | 23,02 | 12,70 | 30,20 | 16,53 |
| Transformer, W 50 | 19,80 | 19,80 ± 0,47 | 119.409 | 26,54 | **11,15** | 29,50 | 17,37 |
| TCN, W 50 | 20,65 | 20,62 ± 1,25 | 102.579 | 22,80 | 19,10 | **29,32** | **16,21** |
| LightGBM, W 30 | 20,17 | 20,16 ± 0,74 | 121.527 | 27,01 | 12,89 | 30,99 | 16,44 |
| LightGBM, W 50 | 20,13 | 20,12 ± 0,72 | 115.158 | 25,59 | 12,64 | 30,77 | 16,63 |
| LightGBM, W 80 (`runs/t2_cv_lgbm_W80`) | 20,82 | 20,81 ± 0,92 | 124.347 | 27,63 | 13,70 | 31,61 | 17,15 |
| GRU awal, sebelum tuning | 24,19 | 24,18 ± 0,93 | 271.342 | 60,30 | 15,11 | 34,43 | 21,93 |

- **GRU W 50 terbaik di kedua metrik**, sehingga dipilih sebagai model final. Selisih RMSE-nya dengan
  Transformer (0,26) lebih kecil dari std antar fold (±0,5–0,7), jadi dari sisi RMSE keduanya setara. Namun skor
  asimetris GRU 19% lebih rendah, dan GRU ±3,5× lebih cepat (419 vs 1.483 detik untuk 5 fold).
- **Dibanding baseline LightGBM**, keunggulan DL dalam RMSE kecil. GRU dan Transformer lebih baik
  (19,54 dan 19,80 vs LightGBM terbaik 20,13), sedangkan TCN sedikit lebih buruk (20,65). Perbedaan yang lebih
  jelas ada di skor asimetris: GRU dan TCN (97.210 dan 102.579) jauh di bawah LightGBM (115.158–124.347),
  sementara Transformer (119.409) setara dengan LightGBM.
- **Window 50 > 30** untuk GRU (19,54 vs 19,89) dan LightGBM (20,13 vs 20,17). W 80 memburuk untuk keduanya
  (LightGBM 20,82; GRU 23,20 di tuning fold 0).
- **Rentang 50–100 adalah titik lemah semua model.** RMSE-nya ±29–31, sementara rentang 0–50 hanya ±11–13
  (TCN pengecualian dengan 19,10). Untuk GRU, 83% skor asimetris (80.394 dari 97.210) berasal dari rentang ini.
  Dugaan kami: label dipotong di 150, sementara awal degradasi tiap mesin berbeda-beda. Mesin yang baru masuk
  fase degradasi sering masih diprediksi mendekati 150 (terlambat, hukuman eksponensial). Pola ini terlihat di
  `uncert_kalibrasi.png`: banyak titik dengan RUL aktual 50–100 diprediksi di atas garis diagonal. Di rentang
  0–50, sinyal degradasi sudah jelas dan prediksi akurat.
- TCN paling buruk di 0–50 (RMSE 19,10, skor 14.102). Model ini paling kurang stabil antar fold (std 1,25).

## 2f. Ketidakpastian dan kapan servis (`task2_eksperimen.py --grup uncert`)

Eksperimen dilakukan di fold 0 (180 mesin validasi, 900 titik potong). Ada 5 GRU (seed 0–4, resep final)
dan tiap model diprediksi 30 kali dengan **MC dropout** (dropout tetap aktif saat inferensi). Prediksi gabungan
memakai μ = rata-rata μᵢ, dengan varians = varians MC dropout + varians antar-seed (ensemble). Ensemble
5 seed menurunkan RMSE dari 20,05 (seed 0) ke 19,73 dan skor dari 18.930 ke 18.609
(`runs/t2_uncert_gru/metrics.json`).

| Interval 90% (fold 0) | Cakupan | Lebar rata-rata |
|---|---|---|
| MC dropout saja (seed 0), μ ± 1,645σ | 57,2% | 24,3 |
| Ensemble saja (std antar 5 seed) | 16,4% | 8,3 |
| Ensemble + MC dropout | 62,2% | 27,3 |
| **Ensemble + MC dropout + split-conformal** | **90,3%** | 56,1 |

- **Interval mentah terlalu sempit.** MC dropout dan ensemble hanya menangkap ketidakpastian *model*. Variasi
  antar mesin yang tidak terlihat di sensor (aleatorik) tidak ikut terukur. Korelasi antara σ dan |galat| hanya
  0,02, artinya σ hampir tidak memberi tahu prediksi mana yang meleset jauh.
- **Kalibrasi split-conformal**: skor r = |y − μ| / σ dihitung pada separuh mesin validasi. Faktor q diambil
  sebagai kuantil 90% (dengan koreksi (n+1)/n) dari r, lalu interval μ ± qσ diuji di separuh lainnya, dua arah.
  Hasilnya q = 3,91 (bukan 1,645) dan cakupan 90,3%, sesuai target. Kurva kalibrasi di semua tingkat nominal
  ditampilkan di `laporan_figs/task2/uncert_kalibrasi.png` dan contoh lintasan mesin di `uncert_lintasan.png`.
  **Keterbatasan**: cakupan per rentang tidak merata, yaitu 91,5% (RUL 0–50), **72,3% (50–100)**, dan
  97,7% (100–150). Interval paling tidak dapat dipercaya justru di rentang yang galatnya paling besar. Selain
  itu, q dihitung dari model fold 0 (720 mesin latih) tetapi dipakai untuk model final (900 mesin).
- **Koreksi bias asimetris** (`geser_bias`): menggeser semua prediksi −15 siklus menurunkan skor asimetris
  fold 0 dari 18.609 ke 7.942, tetapi RMSE naik dari 19,73 ke 22,57. Kedua metrik dinilai di test tersembunyi,
  dan pergeseran sebesar itu dipilih dengan melihat fold yang sama. Karena itu **kami tidak menggeser**
  (geser = 0).

**Aturan servis.** Sebuah titik dianggap "harus diservis" bila RUL aktual ≤ 10 (39 titik di fold 0) dan
"aman" bila RUL > 50 (701 titik). Jumlah dihitung dari `runs/t2_unc_gru_s0/oof.csv.gz`.

| Aturan (servis bila … ≤ T) | T | Tertangkap (RUL ≤ 10) | Alarm palsu (RUL > 50) | Fraksi diservis | Sisa umur terbuang |
|---|---|---|---|---|---|
| prediksi titik | 10 | 33,3% | 0,0% | 2,0% | 6,8 |
| prediksi titik | 20 | 69,2% | 0,0% | 6,9% | 13,0 |
| prediksi titik | 30 | 100% | 0,3% | 13,3% | 16,9 |
| batas bawah 90% | 10 | 100% | 0,6% | 15,8% | 19,4 |
| batas bawah 90% | 20 | 100% | 1,3% | 18,0% | 22,1 |
| batas bawah 90% | 30 | 100% | 1,9% | 21,2% | 24,6 |
| batas bawah 90% | 40 | 100% | 5,1% | 24,6% | 30,0 |

Memakai prediksi titik dengan ambang kecil berbahaya: pada T = 10, dua dari tiga mesin yang hampir rusak
terlewat. Memakai **batas bawah interval 90%** menangkap semua mesin yang hampir rusak bahkan pada T = 10,
dengan biaya lebih banyak servis dan ±20–25 siklus umur terbuang per servis. Di fold 0, prediksi titik ≤ 30
juga menangkap semuanya dengan servis lebih sedikit. Namun ambang itu bergantung pada prediksi titik yang
bias terlambat di rentang 50–100, dan sampelnya kecil (39 titik). Untuk data uji kami menyarankan aturan yang
lebih konservatif: **servis bila batas bawah interval 90% ≤ 30 siklus**. Aturan ini menangkap 100% mesin
hampir rusak di validasi dengan alarm palsu 1,9%. Karena kerusakan mendadak jauh lebih mahal daripada servis
yang sedikit terlalu awal, trade-off ini layak. Ambang T dapat diturunkan bila biaya servis tinggi.

**Model final dan prediksi uji.** Lima GRU (seed 0–4, resep final) dilatih pada seluruh 900 mesin latih
(`runs/t2_full_gru_s{0..4}`, rata-rata prediksi uji per seed 93,71–94,48). Masing-masing diprediksi dengan
MC dropout 30 kali, memakai seed tetap supaya bisa direproduksi. Interval memakai q = 3,91 dari fold 0. Hasilnya
`prediksi_rul.csv`: 300 mesin dengan rata-rata RUL 94,08, dan **24%** mesin disarankan servis sekarang
(`runs/t2_final/metrics.json`). Rincian per mesin beserta batas bawah/atas ada di
`runs/t2_final/prediksi_rul_dengan_interval.csv`.

**Catatan kebocoran dan keterbatasan.**
1. Statistik normalisasi, label, window, dan penanganan spike/NaN semuanya kausal atau hanya dari data latih.
   Data uji hanya dipakai untuk prediksi akhir.
2. Resep dipilih di fold 0, dan fold 0 juga ikut dalam rata-rata CV, sehingga estimasi CV sedikit optimis.
   Analisis ketidakpastian juga memakai fold 0.
3. Validasi memakai 5 titik potong per mesin, sedangkan test memakai 1 titik per mesin, sehingga skor total
   tidak sebanding. Skor per titik (21,60 untuk GRU) lebih tepat untuk memperkirakan skor test per mesin.

## Referensi
- Cho, K. dkk. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine
  Translation. EMNLP.
- Bai, S., Kolter, J. Z., Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent
  Networks for Sequence Modeling. arXiv:1803.01271.
- Vaswani, A. dkk. (2017). Attention Is All You Need. NeurIPS.
- Ke, G. dkk. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
- Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep
  Learning. ICML.
- Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation
  using Deep Ensembles. NeurIPS.
- Angelopoulos, A. N., Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification. arXiv:2107.07511.
- Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation. PHM 2008 (asal skor asimetris dan RUL ber-cap).
