# Task 3: Clustering Material Multimodal Tanpa Label

## Masalahnya

Ada 36.000 sampel material. Masing-masing punya **citra mikrostruktur** grayscale 64×64 dan
**spektrum komposisi** 128 titik. Tidak ada label. Tugasnya mengelompokkan sampel menurut jenis
material, lalu menulis hasilnya ke `cluster.csv`. Dosen menilai dengan ARI terhadap label yang
disembunyikan.

Kesulitannya ada pada data yang penuh **distorsi pengganggu** yang tidak berhubungan dengan material:

| Modalitas | Distorsi (temuan EDA) |
|---|---|
| Citra | Rotasi acak, perbedaan mikroskop (M2 lebih terang, M1 lebih blur), variasi perbesaran |
| Spektrum | Baseline miring/lengkung, spike (22,5% spektrum), pergeseran kecil |

Kalau sampel dikelompokkan langsung, cluster bisa mengikuti mikroskop atau rotasi, bukan material.

## Ide utama

**SimCLR** (Chen dkk., 2020): dua versi augmentasi dari sampel yang sama dipaksa punya embedding mirip.
Augmentasi kami rancang **meniru distorsi pengganggu dari EDA** (rotasi, kecerahan, blur, baseline,
spike). Dengan begitu encoder belajar mengabaikan distorsi, dan yang tersisa di embedding adalah ciri material.

```
citra ────► CNN (SimCLR citra) ────────► 256-d ─► PCA 32 ─► L2 ─┐
                                                               ├─► concat (bobot 0,5 : 0,5) ─► K-Means, k = 8
spektrum ─► 1D-CNN (SimCLR joint) ─────► 256-d ─► PCA 32 ─► L2 ─┘
```

Encoder dilatih dari nol dengan PyTorch. Label dan atribut `mikroskop` **tidak pernah** dipakai untuk
melatih atau mengelompokkan. Atribut mikroskop hanya dipakai untuk memeriksa bahwa cluster tidak mengikutinya.

## Keputusan penting dan alasannya

| Keputusan | Alasan | Bukti |
|---|---|---|
| Augmentasi meniru distorsi | SimCLR hanya bisa menurunkan loss dengan mengabaikan hal yang diubah augmentasi | `task3_ssl.py`, laporan 3b |
| Memilih representasi lewat **kesepakatan lintas modalitas** | Tanpa label, kriteria yang dipakai: cluster dari citra saja dan cluster dari spektrum saja harus sepakat (ARI tinggi). Material memengaruhi keduanya, sedangkan pengganggu hanya memengaruhi salah satunya | `runs/t3_cluster_banding` |
| Citra: SimCLR citra. Spektrum: SimCLR *joint* | Kesepakatan tertinggi. ResNet18 ImageNet (pembanding) jauh lebih rendah | `runs/t3_cluster_banding` |
| Fusi kedua modalitas | Citra saja tidak bisa memisahkan 2 pasang cluster, spektrum saja tidak bisa memisahkan 2 pasang lain. Fusi memisahkan keempatnya | `runs/t3_cluster_final` → `modality` |
| **k = 8** | Silhouette/Davies-Bouldin menunjukkan 5 kelompok kasar. UMAP dan uji belah per modalitas menunjukkan 3 di antaranya terbelah dua. Jadi 5 + 3 = 8 | `runs/t3_cluster_k`, `uji_belah.csv` |
| **K-Means** (bukan GMM/DEC) | Silhouette seri dengan DEC, tetapi stabilitas antar seed lebih tinggi (1,000 vs 0,884). Model deep learning-nya adalah encoder SimCLR, dan K-Means mengelompokkan embedding-nya | `runs/t3_cluster_final` → `metode` |

## Hasil final (k = 8)

Sumber: `runs/t3_cluster_final/metrics.json`

| Metode | Silhouette | Davies-Bouldin | Stabilitas 3 seed (ARI) |
|---|---|---|---|
| **K-Means (final)** | **0,2293** | **1,7824** | **1,0000** |
| GMM | 0,2239 | 1,8152 | 0,8013 |
| DEC | 0,2292 | 1,7838 | 0,8835 |

- Ukuran cluster seimbang (3.656–5.787 sampel).
- NMI cluster terhadap mikroskop ≈ 0, artinya cluster tidak mengikuti mikroskop.
- K-Means dan DEC, dua cara kerja yang berbeda, menghasilkan partisi hampir identik (ARI 0,991).

**Keterbatasan:** pasangan cluster 1 dan 6 paling sulit dipisah: citranya mirip, spektrumnya hanya
berbeda di tinggi relatif dua puncak. Kalau keduanya ternyata satu material, k yang tepat adalah 7.

## File di folder ini

| File | Isi | Poin soal |
|---|---|---|
| `task3_eda.py` | Mengukur distorsi tanpa label (kecerahan per mikroskop, rotasi, baseline, spike) | 3a |
| `task3_ssl.py` | SimCLR citra / spektrum / joint, dan ResNet18 ImageNet sebagai pembanding (`--modal`) | 3b |
| `task3_cluster.py` | `--tahap banding` (pilih representasi), `k` (pilih jumlah cluster), `final` (K-Means vs GMM vs DEC + analisis) | 3c–3f |

## Cara menjalankan

```bash
# angka laporan + cluster.csv dari weights tersimpan (cepat)
python kode/reproduce.py --data data_uts_dl --task 3

# uji cepat di CPU dengan subset data
python kode/task3_cluster/task3_ssl.py --data data_kecil --modal img --smoke

# melatih satu encoder SimCLR, contoh: citra
python kode/task3_cluster/task3_ssl.py --data data_uts_dl --modal img --nama t3_ssl_img_s0
```

Urutan lengkap (dijalankan otomatis oleh `reproduce.py --retrain`): `task3_eda.py` → `task3_ssl.py`
(`img`, `spec`, `joint`, `pretrained`) → `task3_cluster.py --tahap banding` → `--tahap k` → `--tahap final`.

Output: `runs/t3_*/`, `laporan_figs/task3/` (termasuk `umap.png`, `contoh_cluster.png`), `cluster.csv`.
Penjelasan lengkap ada di laporan bab Task 3 (`laporan/bab_task3.md`).

**Pretrained:** ResNet18 bobot ImageNet dari torchvision (He dkk., 2016), hanya sebagai pembanding,
tidak dipakai di model final.
