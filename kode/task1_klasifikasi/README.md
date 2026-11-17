# Task 1: Klasifikasi Cacat Produk Multimodal

## Masalahnya

Setiap unit produk punya **gambar** grayscale 64×64 dan **sinyal sensor** 3 kanal (suhu, getaran,
arus) × 128 titik. Tugasnya menebak 1 dari 7 kelas: `normal`, `gores`, `noda`, `retak`, `gelembung`,
`las_dingin`, `kontaminasi`. Ada 48.000 unit latih dan 16.000 unit uji. Metrik utamanya macro-F1.

Data ini punya tiga kesulitan, dan masing-masing kami tangani:

| Kesulitan | Temuan EDA | Cara menangani |
|---|---|---|
| Sensor sering hilang | Tiap kanal hilang ±6%, 5% unit kehilangan ketiganya | Kanal *mask* + **modality dropout** |
| Label dari inspektur tidak sempurna | 1.241 kandidat salah label (2,6%) | Deteksi out-of-fold, kandidat di bagian latih dibuang |
| Data uji didominasi lini produksi lain | Lini B hanya 25% data latih tetapi 60% data uji | Diukur per lini + skor tertimbang komposisi test |

## Alur pipeline

```
gambar 64×64 ──► CNN 4 blok ──► 16 token ─┐
                                          ├─► cross-attention dua arah ─► MLP ─► 7 kelas
sensor 3×128 ─┬► 1D-CNN 4 blok ─► 8 token ┘
mask 3×128 ───┘   (kanal mask = 1 jika sensor terekam)
```

Semua dilatih dari nol dengan PyTorch, tanpa data atau bobot eksternal.

## Keputusan penting dan alasannya

| Keputusan | Alasan | Bukti |
|---|---|---|
| CNN untuk gambar, 1D-CNN untuk sensor | Cacat adalah pola lokal (goresan, titik, plateau arus), dan bias induktif CNN cocok untuk pola lokal | `task1_model.py` |
| Fusi **cross-attention** | Unggul atas *concat* dan *late fusion* di semua kolom, terutama lini B | `runs/t1_mod_*` |
| Butuh kedua modalitas | `las_dingin` hanya terlihat di sensor (F1 gambar-saja 0,000), sedangkan `kontaminasi`, `noda`, `gores` hanya terlihat di gambar | `runs/t1_mod_*`, `laporan_figs/task1/modality_f1_heatmap.png` |
| **Modality dropout 25%** | Tanpa ini, model fusi lebih buruk daripada model gambar-saja ketika sensor hilang (0,616 vs 0,687). Dengan dropout, saat training sensor sengaja "dimatikan", sehingga cabang gambar belajar mandiri | `runs/t1_miss_drop25_xattn` |
| Buang kandidat salah label | Prediksi out-of-fold 5-fold menemukan label yang bertentangan dengan model yang yakin. Hanya sampel di bagian latih yang dibuang, validasi tidak diubah | `task1_noise.py`, `runs/t1_noise` |
| Loss berbobot 1/√frekuensi | Kelas tidak seimbang (`normal` 50%, `kontaminasi` 5%) | `runs/t1_eda` |
| Normalisasi dari data latih saja | Mencegah kebocoran informasi data uji | `task1_data.py` |
| Model dari epoch terakhir, bukan epoch terbaik | Agar skor validasi tidak bias karena memilih epoch | `task1_train.py` |

**Yang dicoba tetapi tidak dipakai.** Lima teknik domain shift (normalisasi per lini, augmentasi
fotometrik, bobot ulang lini, gabungan keduanya, augmentasi garis halus hasil analisis Fourier)
tidak menaikkan skor lini B. Hasil ini dilaporkan apa adanya di laporan bagian 1e.

## Hasil model final (validasi, 3 seed)

Sumber: `runs/t1_tabel/final_mean_std.csv`

| Ukuran | mean ± std |
|---|---|
| macro-F1 | **0,8700 ± 0,0012** |
| macro-F1 lini A / lini B | 0,8992 / 0,7798 |
| macro-F1 tertimbang komposisi test (perkiraan skor uji) | 0,8286 ± 0,0015 |
| macro-F1 jika semua sensor hilang | 0,6704 ± 0,0023 |

Prediksi uji (`prediksi_klasifikasi.csv`) adalah rata-rata probabilitas dari 3 seed.

**Keterbatasan utama:** lini B (0,780). Gambar lini B punya garis halus berarah yang mirip goresan
dan retakan, dan belum teratasi oleh teknik yang kami uji.

## File di folder ini

| File | Isi | Poin soal |
|---|---|---|
| `task1_eda.py` | Eksplorasi: contoh gambar, sinyal per kelas, sensor hilang, perbedaan lini | 1a |
| `task1_data.py` | Muat data, normalisasi (statistik latih), split validasi stratified per kelas × lini | semua |
| `task1_model.py` | Encoder gambar, encoder sinyal, tiga strategi fusi | 1b |
| `task1_train.py` | Satu skrip training/evaluasi untuk semua eksperimen | semua |
| `task1_noise.py` | Deteksi kandidat salah label (out-of-fold 5-fold) | 1d |
| `task1_tekstur.py` | Analisis Fourier tekstur lini A vs B | 1e |
| `task1_eksperimen.py` | Menjalankan grup eksperimen dan membuat tabel/figur | 1b–1g |

## Cara menjalankan

```bash
# angka laporan + prediksi dari weights tersimpan (cepat)
python kode/reproduce.py --data data_uts_dl --task 1

# uji cepat di CPU dengan subset data
python kode/task1_klasifikasi/task1_train.py --data data_kecil --smoke

# satu eksperimen penuh, contoh: fusi cross-attention
python kode/task1_klasifikasi/task1_train.py --data data_uts_dl --modal fusion --fusion xattn --nama t1_xattn --seed 0

# satu grup eksperimen, contoh: perbandingan modalitas
python kode/task1_klasifikasi/task1_eksperimen.py --data data_uts_dl --grup modality
```

Urutan lengkap (dijalankan otomatis oleh `reproduce.py --retrain`): `task1_eda.py` → `task1_tekstur.py`
→ grup `sanity` → `modality` → `task1_noise.py` → grup `noise` → `domain` → `missing` → `final`.

Output: `runs/t1_*/` (`config.json`, `metrics.json`, kurva), `laporan_figs/task1/`, `prediksi_klasifikasi.csv`.
Penjelasan lengkap ada di laporan bab Task 1 (`laporan/bab_task1.md`).
