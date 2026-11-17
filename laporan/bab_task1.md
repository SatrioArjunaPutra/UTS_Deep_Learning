# Task 1 — Klasifikasi Multimodal Cacat Produk

Penulis bab: Satrio Arjuna Putra. Semua angka berasal dari `runs/<run>/metrics.json` (tabel lengkap:
`docs/tabel_laporan.md`, dibuat otomatis oleh `kode/tabel_laporan.py`). Validasi = 9.600 unit (20%,
stratified per kelas × lini, seed split tetap 0) sehingga semua eksperimen dibandingkan di himpunan yang sama.

## 1a. Eksplorasi

Data latih 48.000 unit dan data uji 16.000 unit; tiap unit punya gambar grayscale 64×64 dan sinyal
sensor 3 kanal (suhu, getaran, arus) × 128 titik (`runs/t1_eda/metrics.json`, Gambar
`laporan_figs/task1/eda_contoh_gambar.png`, `eda_sinyal_per_kelas.png`).

- **Kelas tidak seimbang** (`eda_distribusi.png`): normal 23.888 unit (49,8%), kontaminasi hanya
  2.357 (4,9%). Karena itu loss memakai bobot kelas 1/√frekuensi dan metrik utama macro-F1.
- **Sensor hilang per kanal secara utuh**: tidak ada kanal yang hilang sebagian (fraksi 0,0). Tiap kanal
  hilang ±5,9%; 4,9% unit latih kehilangan ketiga kanal (test 5,1%). Persentasenya merata antar kelas
  (4,2–5,6%) dan antar lini, jadi hilangnya sensor tidak membocorkan label. Masukan sensor diberi
  3 kanal mask (1 = terekam) dan NaN diisi 0 setelah normalisasi.
- **Sinyal per kelas**: selisih rata-rata sinyal terhadap kelas normal (dalam satuan std) besar hanya
  untuk `las_dingin` pada arus (2,84) dan `gelembung` pada suhu (0,25); kelas lain < 0,04. `retak` sedikit
  menaikkan std getaran (0,805 vs ±0,786). Dugaan awal: `las_dingin` butuh sensor, kelas lain butuh gambar.
- **Perbedaan train–test per lini** (`eda_shift_lini.png`): lini B hanya 25,1% data latih tetapi
  60,0% data uji (mesin M07–M12 muncul ±2,4× lebih sering di test). Gambar lini B lebih gelap
  (kecerahan rata-rata 111,7 vs 139,8) dan kontrasnya lebih tinggi (17,1 vs 12,8); suhu lini B ±2 °C lebih
  tinggi. Distribusi kelas per lini hampir sama, jadi pergeserannya ada di *input* (covariate shift), bukan di label.
  Akibatnya skor validasi biasa (75% lini A) terlalu optimis untuk data uji; kami juga melaporkan
  **macro-F1 tertimbang komposisi test** (sampel validasi diberi bobot p_test(lini)/p_train(lini)).
  Bobot ini hanya memakai atribut `lini` di meta test, tanpa label.

## 1b. Arsitektur (`kode/task1_model.py`)

- **Encoder gambar**: CNN 4 blok (2× conv3×3-BN-ReLU + maxpool), 64×64 → 4×4×256. Cacat adalah pola
  lokal, sehingga bias induktif CNN cocok; dilatih dari nol (tanpa data/bobot eksternal).
- **Encoder sinyal**: 1D-CNN 4 blok (kernel 5), masukan 6 kanal (3 nilai + 3 mask), 128 → 8 token × 256.
- **Tiga strategi fusi** (dibandingkan pada konfigurasi training identik, 25 epoch, AdamW + OneCycle):
  1. *concat* (intermediate fusion): [v_gambar, v_sensor × ada, mask] → MLP.
  2. *late*: kepala klasifikasi terpisah; logit = l_gambar + a_k · ada · l_sensor (a_k per kelas dipelajari).
  3. *xattn*: cross-attention dua arah antara 16 token gambar dan 8 token sinyal, plus satu token
     "kosong" yang dipelajari agar attention tetap terdefinisi saat sensor hilang.

| Model (validasi) | Param | Macro-F1 | Lini A | Lini B | Tertimbang test |
|---|---|---|---|---|---|
| gambar saja | 1,24 jt | 0,687 | 0,721 | 0,584 | 0,639 |
| sensor saja | 0,72 jt | 0,458 | 0,471 | 0,417 | 0,439 |
| fusi concat | 1,96 jt | 0,837 | 0,870 | 0,736 | 0,790 |
| fusi late | 1,96 jt | 0,830 | 0,862 | 0,729 | 0,784 |
| **fusi xattn** | 2,49 jt | **0,844** | **0,873** | **0,754** | **0,803** |

Cross-attention terbaik di semua kolom, terutama di lini B (+0,018 dari concat), sehingga dipakai sebagai
dasar eksperimen 1d–1g. Selisih antar fusi kecil (≤ 0,014), jadi kesimpulan "xattn > concat" perlu
dibaca bersama variasi antar seed di 1g.

## 1c. Nilai tiap modalitas (Gambar `laporan_figs/task1/modality_f1_heatmap.png`)

| F1 per kelas | normal | gores | noda | retak | gelembung | las_dingin | kontaminasi |
|---|---|---|---|---|---|---|---|
| gambar saja | 0,854 | 0,769 | 0,881 | 0,650 | 0,716 | **0,000** | 0,941 |
| sensor saja | 0,653 | 0,107 | 0,097 | 0,608 | 0,736 | **0,929** | 0,074 |
| fusi xattn | 0,908 | 0,773 | 0,877 | 0,689 | 0,790 | 0,936 | 0,933 |

- **Hanya sensor**: `las_dingin` (F1 gambar 0,000 vs sensor 0,929), sesuai EDA (plateau arus lebih rendah).
- **Hanya gambar**: `kontaminasi`, `noda`, `gores` (F1 sensor ≤ 0,11).
- **Samar di keduanya**: `gelembung` (0,716 / 0,736 → fusi 0,790) dan `retak` (0,650 / 0,608 → fusi 0,689);
  di dua kelas ini fusi memberi kenaikan terbesar karena tiap modalitas membawa informasi sebagian.
- `normal` naik dari 0,854 (gambar) ke 0,908 (fusi) karena sensor membantu menolak `las_dingin`.

## 1d. Label tidak sempurna

**Deteksi.** Prediksi out-of-fold 5-fold (StratifiedKFold per kelas×lini, model xattn 15 epoch,
modality dropout 0,1; `kode/task1_noise.py`) memberi macro-F1 OOF 0,865. Sampel ditandai *kemungkinan salah label* bila
p(label tercatat) < 0,05 sementara model > 0,80 yakin pada kelas lain: **1.241 unit (2,6%)**
(`runs/t1_noise/mungkin_salah_label.csv`, contoh gambar `laporan_figs/task1/noise_contoh.png`,
matriks `noise_matriks.png`). Proporsi per lini (980 A / 261 B) sebanding dengan komposisi latih.

Pasangan terbanyak (label → prediksi OOF): normal→las_dingin 161, retak→normal 131, gores→normal 121,
gelembung→noda 121, noda→gelembung 107, retak→gores 104, gores→retak 95, las_dingin→normal 92.
Dua pola terlihat: (1) **normal ↔ las_dingin** — cacat yang hanya terlihat di sensor arus; inspektur yang
menilai secara visual wajar melewatkannya; (2) pasangan simetris **gelembung ↔ noda** dan **retak ↔ gores**
— kelas yang mirip secara visual, sehingga sebagian kasus lebih tepat disebut ambigu daripada salah tulis.

**Teknik tahan noise** (xattn, konfigurasi lain identik; sampel hanya dibuang dari data *latih*,
himpunan validasi tidak diubah):

| Run | Macro-F1 | Lini A | Lini B | Tertimbang test |
|---|---|---|---|---|
| baseline | 0,8440 | 0,8731 | 0,7539 | 0,8025 |
| label smoothing 0,1 | 0,8436 | 0,8726 | 0,7561 | 0,8030 |
| buang 1.241 kandidat salah label | **0,8521** | **0,8827** | **0,7593** | **0,8091** |

Label smoothing tidak mengubah skor; membuang kandidat salah label menaikkan macro-F1 +0,008. Catatan
jujur: label validasi juga berasal dari inspektur (ikut noisy), dan ini hasil satu seed, sehingga kenaikan
diuji ulang dengan 3 seed pada model final (1g).

## 1e. Domain shift

**Pengukuran per lini.** Pada model xattn, macro-F1 lini A 0,873 tetapi lini B hanya 0,754. F1 per kelas
per lini (`runs/t1_tabel/f1_per_kelas_per_lini.csv`) menunjukkan penurunan terpusat pada cacat **visual
yang tipis**: gores 0,832 → 0,588, retak 0,750 → 0,492, noda 0,915 → 0,768, sedangkan kelas yang dikenali
dari sensor stabil (las_dingin 0,939 → 0,928).

**Mengapa.** Analisis Fourier 2.000 gambar `normal` data latih per lini (`kode/task1_tekstur.py`,
`runs/t1_tekstur/metrics.json`, Gambar `laporan_figs/task1/domain_fourier.png`) menunjukkan gambar lini B
punya energi tambahan pada cincin frekuensi 13–18 siklus/64 px (periode ±4 px): median porsi energi cincin
9,8% vs 2,2% di lini A (energi absolut 27,1 vs 3,1). Di dalam satu gambar lini B, 65% energi cincin berada
dalam ±20° dari arah puncaknya (lini A 27%, mendekati merata), tetapi arah puncak itu tersebar merata antar
gambar (histogram 6 bin 0–180°: 375/365/292/357/301/310). Jadi gambar lini B memuat **garis halus berarah
dengan arah acak per gambar** — pola yang mirip goresan dan retakan tipis, sehingga menyamarkan keduanya. Kecerahan (111,7 vs 139,8) dan kontras berbeda juga, tetapi itu mudah dinormalisasi.

**Teknik yang diuji** (xattn, satu seed, validasi sama):

| Run | Macro-F1 | Lini A | Lini B | Tertimbang test |
|---|---|---|---|---|
| baseline | 0,8440 | 0,8731 | **0,7539** | **0,8025** |
| normalisasi per lini (statistik latih per lini) | 0,8402 | 0,8720 | 0,7439 | 0,7955 |
| augmentasi fotometrik + offset suhu | 0,8462 | 0,8798 | 0,7439 | 0,7988 |
| bobot ulang lini (p_test/p_train) | 0,8380 | 0,8674 | 0,7490 | 0,7967 |
| per lini + augmentasi | 0,8371 | 0,8717 | 0,7329 | 0,7886 |
| augmentasi garis halus lini B (p = 0,5) | 0,8379 | 0,8677 | 0,7469 | 0,7958 |

Empat teknik pertama hanya menyamakan kecerahan, kontras, suhu, atau bobot, sehingga tidak menyentuh
penyebab utama (garis halus); tidak satu pun menaikkan lini B. Teknik terakhir dirancang dari analisis
Fourier di atas: noise yang disaring ke cincin frekuensi yang sama dan ke irisan arah (Gaussian sd 22°)
di sekitar arah acak per gambar ditambahkan ke 50% gambar latih, dengan kekuatan disetel dari selisih
energi cincin lini B − A di data latih (tambahan std ±4,8 piksel). Garis tiruan ini meniru lini B pada
ketiga ukuran (`runs/t1_tekstur`, kolom "lini A + garis tiruan"; contoh visual di `domain_fourier.png`).
Tujuannya agar model belajar mengenali goresan/retakan meskipun tertimpa garis halus lini B.

**Hasil jujur: tidak ada teknik domain yang menaikkan lini B.** Augmentasi garis halus pun menurunkan lini B
(0,7469 vs 0,7539). Dugaan kami: menambah garis tiruan membuat gambar lini A ikut lebih sulit tanpa mengajari
model cara membedakan garis lini B dari goresan asli, karena keduanya memang mirip di skala 64×64. Karena itu
tidak ada teknik domain yang dipakai di model final. Yang ternyata menaikkan lini B adalah modality dropout
(1f: lini B 0,7539 → 0,7824), yang memaksa cabang gambar belajar lebih kuat. Sisa celah lini A–B tetap
menjadi keterbatasan utama model untuk data uji (60% lini B).

## 1f. Modalitas hilang

Pada model fusi yang dilatih tanpa modality dropout, menghapus semua sensor di validasi menurunkan
macro-F1 xattn dari 0,844 ke 0,616 — **lebih rendah daripada model gambar saja (0,687)**, padahal
batas atas tanpa sensor kira-kira setara model gambar saja (`las_dingin` mustahil dikenali). Jadi model
fusi biasa belum "masuk akal" saat sensor tidak terekam: ia terlalu bergantung pada sensor.

**Modality dropout** (`--mod_drop 0.25`): saat training, seluruh sensor sebuah unit dihapus (mask = 0) dengan
peluang 25%, dan satu kanal dihapus dengan peluang 5%. Sensor yang dihapus benar-benar tidak terlihat
model: di encoder sinyal nilai dikalikan mask, dan cabang sensor tiap fusi dimatikan bila tidak ada kanal.

| Run (xattn, validasi) | Macro-F1 | Lini A | Lini B | Tertimbang test | Tanpa sensor (semua dihapus) | Sensor hilang asli (476 unit) |
|---|---|---|---|---|---|---|
| baseline | 0,8440 | 0,8731 | 0,7539 | 0,8025 | 0,6164 | 0,6217 |
| modality dropout 25% | **0,8692** | **0,8969** | **0,7824** | **0,8295** | **0,6705** | **0,6841** |
| (acuan) model gambar saja | 0,6872 | 0,7210 | 0,5835 | 0,6394 | 0,6872 | 0,6881 |

Tanpa sensor, model dengan modality dropout mendekati model khusus gambar (0,671 vs 0,687) — batas atas
yang wajar karena `las_dingin` hanya terlihat di sensor — jadi prediksinya kini masuk akal saat sensor hilang.
Yang tidak diduga: dengan sensor lengkap pun macro-F1 naik 0,025 dan lini B naik 0,029. Karena sensor sering
"dimatikan", cabang gambar dipaksa belajar sendiri, sehingga cacat visual ikut membaik (lini B: gores
0,588 → 0,614, retak 0,492 → 0,578; `runs/t1_tabel/f1_per_kelas_per_lini.csv`). Modality dropout 25% dipakai
di model final.

## 1g. Training dan model final

- **Sanity check**: model fusi xattn yang sama dengan model final dilatih pada satu batch 64 sampel;
  setelah 300 langkah loss turun dari 1,96 ke 5,6·10⁻⁵ dan akurasi 1,000 (`runs/t1_sanity_overfit/metrics.json`
  dan `history.json`, kurva `runs/t1_sanity_overfit/kurva_overfit.png`), menandakan pipeline data–model–loss benar dan model cukup kapasitasnya.
- Model yang dievaluasi selalu model **epoch terakhir** (jadwal OneCycle), bukan epoch terbaik di validasi,
  agar skor validasi tidak bias oleh pemilihan epoch.
- **Konfigurasi final** (dipilih di validasi dengan aturan yang ditetapkan sebelum melihat hasil: macro-F1
  tertimbang komposisi test): fusi xattn + buang 1.241 kandidat salah label (1d) + modality dropout 25% (1f);
  tidak ada teknik domain (1e) karena tidak satu pun menaikkan skor tertimbang. 25 epoch, AdamW (lr 2·10⁻³,
  wd 5·10⁻⁴), OneCycle, batch 256; dilatih dengan **3 seed (0, 1, 2)** pada split validasi yang sama.

| Model final, validasi (`runs/t1_tabel/final_mean_std.csv`) | mean ± std (3 seed) |
|---|---|
| macro-F1 | **0,8700 ± 0,0012** |
| macro-F1 lini A / lini B | 0,8992 ± 0,0022 / 0,7798 ± 0,0031 |
| macro-F1 tertimbang komposisi test (60% lini B) | 0,8286 ± 0,0015 |
| macro-F1 tanpa sensor (semua sensor dihapus) | 0,6704 ± 0,0023 |
| F1 normal / gores / noda / retak | 0,922 / 0,811 / 0,886 / 0,749 |
| F1 gelembung / las_dingin / kontaminasi | 0,837 / 0,941 / 0,944 |

- **Variasi antar seed kecil** (std macro-F1 0,0012), jadi selisih antar eksperimen satu-seed di 1b–1f yang
  lebih kecil dari ±0,003 sebaiknya dibaca sebagai setara. Contohnya, menambah "buang kandidat salah label" di
  atas modality dropout tidak menambah skor (seed 0: 0,8687 vs modality dropout saja 0,8692).
- **Kurva pembelajaran** (`laporan_figs/task1/final_kurva.png`): loss validasi bergejolak di 5 epoch awal
  (fase learning rate tinggi OneCycle), lalu stabil di ±0,5 sementara loss latih terus turun ke ±0,26; macro-F1
  validasi naik stabil dan mendatar di ±0,87 sejak epoch ±18, tanpa tanda overfitting yang menurunkan skor.
- **Confusion matrix** (`final_confusion.png`, dijumlah atas 3 seed): kesalahan terbesar adalah cacat yang
  diprediksi `normal` (terutama retak dan gores) dan retak ↔ gores — kelas yang sama dengan pasangan kandidat
  salah label di 1d dan kelas yang paling terdampak lini B di 1e.
- **Prediksi data uji**: rata-rata probabilitas 3 seed (ensemble) → `prediksi_klasifikasi.csv`
  (16.000 baris; lolos `kode/cek_submission.py --data data_uts_dl`: kolom, nama kelas, dan himpunan ID persis).

**Ringkasan Task 1.** Fusi cross-attention + modality dropout memberi macro-F1 validasi 0,870 ± 0,001. Kelemahan
utama ada di lini B (0,780), yang mendominasi data uji; garis halus lini B mirip goresan/retakan dan belum
teratasi oleh teknik domain yang kami uji, sehingga skor uji kemungkinan lebih dekat ke estimasi tertimbang
(±0,83) daripada skor validasi biasa.
