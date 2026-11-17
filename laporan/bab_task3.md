# Task 3 — Clustering Multimodal: Jenis Material

Penulis bab: Satrio Arjuna Putra. Angka dari `runs/t3_eda`, `runs/t3_ssl_*`, dan `runs/t3_cluster_*`
(`metrics.json`; tabel lengkap di `docs/tabel_laporan.md`).
Task 3 tidak memiliki label; atribut `mikroskop` di meta **tidak** dipakai untuk melatih atau
mengelompokkan, hanya untuk memeriksa apakah representasi/cluster mengikuti faktor pengganggu.

## 3a. Eksplorasi distorsi (`kode/task3_eda.py`, Gambar `laporan_figs/task3/eda_contoh.png`, `eda_distorsi.png`)

36.000 sampel: citra mikrostruktur grayscale 64×64 dan spektrum 128 titik.

**Citra**
- **Rotasi**: histogram orientasi tekstur (10 bin, 0–180°) hampir merata (1.279–2.486 sampel per bin);
  55,8% sampel memiliki tekstur berarah kuat. Orientasi tersebar ke segala arah, jadi rotasi adalah
  distorsi akuisisi, bukan ciri material.
- **Perbedaan mikroskop**: 60,1% sampel dari M1 (21.636) dan 39,9% dari M2 (14.364). M2 lebih terang
  (kecerahan rata-rata 146,1 vs 127,8) dan lebih tajam (energi Laplacian 20,7 vs 14,7; M1 lebih blur),
  sedangkan kontras (19,7 vs 19,9) dan frekuensi tekstur dominan (5,99 vs 5,97) hampir sama.
- **Skala & kecerahan**: variasi perbesaran dan kecerahan antar sampel (lihat `eda_contoh.png`).

**Spektrum**
- **Spike**: 22,5% spektrum memiliki spike (rata-rata 0,26 spike per spektrum), merata di M1/M2
  (0,259 vs 0,253).
- **Baseline**: kemiringan baseline bervariasi lebar (persentil 5–95%: −0,92 s.d. 0,50).
- **Pergeseran/puncak**: tiap spektrum memiliki beberapa puncak (pengamatan visual `eda_contoh.png`); posisi puncak tertinggi tersebar di indeks 17–80 (persentil 5–95%).
  Sebagian variasi posisi ini mungkin ciri material, sehingga augmentasi pergeseran dibuat kecil (±6 titik).
- Baseline dan spike tidak berbeda antar mikroskop (baseline −0,088 vs −0,092), jadi faktor mikroskop
  terutama memengaruhi citra.

**Kesimpulan distorsi dominan**: citra — rotasi dan perbedaan mikroskop (kecerahan + blur);
spektrum — baseline miring/lengkung dan spike.

## 3b. Representasi tanpa label (`kode/task3_ssl.py`)

**Metode utama: SimCLR** (Chen dkk., 2020). Dua augmentasi acak dari sampel yang sama dipaksa
berdekatan di ruang embedding (pasangan positif), sampel lain di batch menjadi negatif (loss NT-Xent,
τ = 0,2, batch 512, 100 epoch untuk ketiga varian). Model hanya bisa meminimalkan loss dengan mengabaikan hal yang diubah
augmentasi, sehingga **augmentasi dipilih untuk meniru distorsi pengganggu dari 3a**, bukan perbedaan material:

| Modalitas | Augmentasi | Alasan (temuan 3a) |
|---|---|---|
| Citra | rotasi bebas 0–360° + flip | orientasi tekstur merata → rotasi bukan ciri material |
| Citra | skala 0,75–1,33 + geser ±7,5% lebar citra | variasi perbesaran |
| Citra | kecerahan ±0,6 std, kontras ×0,6–1,4 | M2 lebih terang dari M1 |
| Citra | blur Gaussian σ 0,3–1,5 (50% sampel) + noise | M1 lebih blur dari M2 |
| Spektrum | geser ±6 titik | pergeseran kalibrasi (dibuat kecil agar posisi puncak tetap bermakna) |
| Spektrum | baseline acak: offset + kemiringan + lengkung | kemiringan baseline bervariasi lebar |
| Spektrum | spike sintetis 0–2 titik (amplitudo 2–8) | 22,5% spektrum punya spike |
| Spektrum | skala amplitudo ×0,8–1,2 + noise kecil | variasi intensitas pengukuran |

**Encoder**: CNN 4 blok untuk citra (64×64 → 256-d, global average pooling) dan 1D-CNN 4 blok untuk
spektrum (128 → 256-d); keduanya dilatih dari nol. Embedding = keluaran encoder sebelum projection head.
Embedding citra dirata-rata atas 8 transformasi dihedral (rotasi 90° × flip) agar lebih invarian rotasi.

**Varian yang dibandingkan**
1. `img`: SimCLR citra saja.
2. `spec`: SimCLR spektrum saja.
3. `joint`: kedua encoder dilatih bersama dengan NT-Xent citra + NT-Xent spektrum + kontras lintas
   modalitas (citra_i ↔ spektrum_i, seperti CLIP, bobot 0,5). Jenis material memengaruhi kedua modalitas,
   sedangkan pengganggu (mikroskop, rotasi, baseline) hanya memengaruhi salah satunya; menyelaraskan kedua
   modalitas mendorong embedding menangkap faktor bersama, yaitu material.
4. `pretrained`: ResNet18 bobot ImageNet dari torchvision (He dkk., 2016;
   https://download.pytorch.org/models/resnet18-f37072fd.pth), tanpa fine-tuning, sebagai pembanding.

**Pemilihan representasi tanpa label** (`runs/t3_cluster_banding/banding_representasi.csv`). Label material
tidak ada, jadi representasi dinilai dari **kesepakatan lintas modalitas**: K-Means dijalankan terpisah pada
embedding citra dan embedding spektrum, lalu dihitung ARI di antara keduanya. Jenis material memengaruhi kedua
modalitas, sedangkan pengganggu (mikroskop, rotasi, baseline) hanya memengaruhi salah satunya; representasi yang
lebih menangkap material menghasilkan kesepakatan lebih tinggi (kriteria ini tertulis di docstring
`task3_cluster.py` sebelum run). Pasangan yang dipilih: rata-rata ARI pada k = 6 dan k = 10 tertinggi.

| Citra | Spektrum | ARI k=6 | ARI k=10 | NMI citra–mikroskop (k=6 / 10) |
|---|---|---|---|---|
| SimCLR citra | SimCLR spektrum | 0,5955 | 0,3636 | 0,0000 / 0,0076 |
| **SimCLR citra** | **SimCLR joint (spektrum)** | **0,5955** | **0,3639** | 0,0000 / 0,0076 |
| SimCLR citra | klasik (median + baseline + PCA) | 0,6407 | 0,3125 | 0,0000 / 0,0076 |
| joint (citra) | joint (spektrum) | 0,5957 | 0,3570 | 0,0000 / 0,0067 |
| ResNet18 ImageNet | joint (spektrum) | 0,4700 | 0,3362 | 0,0066 / 0,0138 |

- Encoder citra SimCLR jelas lebih baik daripada ResNet18 ImageNet (bobot ImageNet tidak mengenal tekstur
  mikrostruktur dan tidak invarian rotasi) dan sedikit lebih baik daripada encoder citra hasil `joint`.
- Untuk spektrum, SimCLR spektrum-saja dan SimCLR `joint` praktis setara (selisih 0,0003 di k = 10); aturan memilih
  `joint`. Pembanding klasik unggul di k = 6 tetapi kalah di k = 10.
- NMI terhadap mikroskop ≈ 0: augmentasi kecerahan/blur berhasil membuat embedding tidak peka mikroskop.

**Representasi final**: citra dari `t3_ssl_img_s0`, spektrum dari `t3_ssl_joint_s0`.

## 3c. Dua modalitas dan fusi

Tiap embedding distandardisasi → PCA 32 dimensi → dinormalisasi L2, lalu digabung
`[√w·citra, √(1−w)·spektrum]` dengan w = 0,5 (tiap blok bernorma 1, sehingga bobot kedua modalitas setara).

Hasil pada k = 8 (`runs/t3_cluster_final/metrics.json` → `modality`; Gambar `modality_kontingensi.png`, `umap.png`):

| Ruang | Silhouette | ARI terhadap cluster fusi |
|---|---|---|
| citra saja | 0,2998 | 0,6926 |
| spektrum saja | 0,2524 | 0,7398 |
| ARI citra-saja vs spektrum-saja | | 0,4315 |

**Apa yang hilang jika hanya satu modalitas** (uji belah di 3e dan `contoh_cluster.png`):
- **Hanya citra**: pasangan cluster 2/7 dan 1/6 bertekstur mirip (seluler/berbutir dan garis halus) dan hanya
  berbeda di spektrum, sehingga masing-masing tergabung menjadi satu cluster.
- **Hanya spektrum**: pasangan 0/4 (garis lurus halus vs garis bergelombang kasar) dan 3/5 (bercak gelap vs sel
  terang) memiliki spektrum yang sama, sehingga tidak bisa dipisahkan tanpa citra.
- Fusi memisahkan keempat pasangan itu; inilah alasan ARI citra-saja vs spektrum-saja hanya 0,4315.

## 3d. Clustering: K-Means vs GMM vs DEC

Pada ruang fusi, k = 8 (`runs/t3_cluster_final/metrics.json` → `metode`, `ARI_antar_metode`):

| Metode | Silhouette | Davies-Bouldin | Stabilitas 3 seed (ARI) | NMI mikroskop |
|---|---|---|---|---|
| **K-Means** | **0,2293** | **1,7824** | **1,0000** | 0,0000 |
| GMM (kovarians penuh, PCA 16) | 0,2239 | 1,8152 | 0,8013 | 0,0000 |
| DEC (Xie dkk., 2016; varian IDEC) | 0,2292 | 1,7838 | 0,8835 | 0,0001 |

- **DEC**: autoencoder MLP (64→256→128→16) dilatih rekonstruksi, pusat awal dari K-Means di ruang laten, lalu
  meminimalkan KL(P‖Q) dengan distribusi target yang dipertajam (+0,1 × loss rekonstruksi agar struktur lokal
  terjaga). DEC berhenti setelah 4 iterasi; hasilnya hampir sama dengan inisialisasinya (ARI 0,9942) dan dengan
  K-Means final (ARI 0,9910). Di ruang latennya sendiri cluster DEC jauh lebih rapat (silhouette 0,5105), tetapi
  stabilitas antar seed lebih rendah daripada K-Means.
- **GMM** menggabungkan satu pasangan citra-saja (satu komponen berisi 8.637 sampel) dan membelah pasangan lain;
  ARI terhadap K-Means hanya 0,8164.
- **Metode final**: aturan `--metode auto` (silhouette fusi tertinggi dibulatkan 3 desimal, seri → stabilitas)
  memilih **K-Means** (silhouette seri dengan DEC, stabilitas 1,0000 vs 0,8835). Model deep learning-nya adalah
  encoder SimCLR; K-Means mengelompokkan embedding-nya. `cluster.csv` = K-Means k = 8.

## 3e. Jumlah cluster

Pemindaian k = 2–20 pada ruang fusi (`runs/t3_cluster_k/metrics.json`, Gambar `pilih_k.png`):

| k | Silhouette 64-d | Davies-Bouldin | Stabilitas | Silhouette UMAP | ARI citra vs spektrum |
|---|---|---|---|---|---|
| 4 | 0,2995 | 1,4606 | 1,0000 | 0,6110 | 0,8007 |
| **5** | **0,3155** | **1,3319** | 0,9343 | 0,6544 | **0,8344** |
| 6 | 0,3076 | 1,3767 | 0,8395 | 0,7449 | 0,5955 |
| 7 | 0,2574 | 1,6593 | 0,7768 | **0,8442** | 0,5366 |
| **8** | 0,2293 | 1,7824 | 0,8940 | 0,7872 | 0,4315 |
| 10 | 0,1998 | 1,9042 | 0,9353 | 0,6513 | 0,3639 |

Kriteria memberi jawaban berbeda karena struktur datanya **bertingkat**:

1. **Silhouette (maks) dan Davies-Bouldin (min) di ruang 64-d → k = 5**: ada 5 kelompok kasar yang terpisah jelas.
   BIC GMM terus turun sampai k = 20 (siku di k = 7), jadi BIC tidak informatif di sini.
2. **UMAP**: silhouette di ruang UMAP 2-D maksimum di k = 7, dan graf kNN pada UMAP terdiri dari 7 komponen
   terpisah (sama untuk 10 dan 30 tetangga). Dua dari 5 kelompok kasar ternyata terdiri dari dua pulau.
3. **Uji belah per modalitas** (`runs/t3_cluster_k/uji_belah.csv`): tiap kelompok kasar diproyeksikan ke arah
   2-means di ruang citra dan ruang spektrum secara terpisah, lalu dihitung rasio lembah histogram proyeksinya
   (≈ 0 = dua puncak terpisah, ≈ 1 = satu gumpalan; ambang 0,3).

| Kelompok kasar | n | Lembah citra | Lembah spektrum | Hasil |
|---|---|---|---|---|
| 0 | 10.043 | 1,000 | **0,004** | terbelah, hanya di spektrum → cluster final 2 + 7 |
| 1 | 8.685 | 1,000 | **0,202** | terbelah lemah, di spektrum → cluster final 1 + 6 |
| 2 | 3.656 | 0,972 | 0,440 | satu gumpalan → cluster final 3 |
| 3 | 4.979 | 1,000 | 0,416 | satu gumpalan → cluster final 5 |
| 4 | 8.637 | **0,003** | 0,502 | terbelah, hanya di citra → cluster final 0 + 4 |

Jadi **5 kelompok kasar + 3 yang terbelah = k = 8**. Silhouette 64-d tidak melihatnya karena perbedaan yang hanya
ada di satu modalitas "terencerkan" setengahnya setelah concat, sementara silhouette/DB menyukai cluster bulat yang
terpisah jauh. Pemeriksaan silang: pada k = 8, K-Means dan DEC (dua cara kerja berbeda) menghasilkan partisi yang
hampir identik (ARI 0,9910), dan ke-8 cluster berukuran seimbang (3.656–5.787 sampel).

Catatan proses: kami mula-mula hanya memakai kriteria 64-d (k = 5). UMAP (3f) memperlihatkan pulau-pulau yang
tergabung, sehingga kriteria UMAP dan uji belah ditambahkan ke `kode/task3_cluster.py` lalu tahap `k` dan `final`
dijalankan ulang. Tidak ada label maupun atribut (mikroskop) yang dipakai dalam keputusan ini.

## 3f. Analisis: UMAP dan cluster yang sulit dipisah

Gambar `umap.png` (UMAP ruang fusi, 30 tetangga) dan `contoh_cluster.png` (6 citra + median spektrum per cluster).

- **Pasangan tersulit: cluster 1 vs 6** (jarak pusat 0,648). Citranya sama-sama garis halus berarah; spektrumnya
  punya tiga puncak di posisi yang sama dan hanya **tinggi relatif dua puncak tengah yang bertukar**. Di UMAP
  keduanya menempel menjadi satu pulau dengan silhouette ≈ 0 di batasnya, dan uji belahnya paling lemah
  (lembah 0,202). Silhouette per cluster: 0,2030 dan 0,1836.
- **Cluster 2 vs 7** (jarak pusat terdekat, 0,538; silhouette terendah 0,1635 dan 0,1480): tekstur
  seluler/berbutir yang mirip, dibedakan oleh spektrum. Di ruang 64-d keduanya berdekatan, tetapi di spektrum
  terpisah tegas (lembah 0,004), sehingga UMAP menampilkannya sebagai dua pulau.
- **Cluster 0 vs 4 dan 3 vs 5**: spektrum identik, dipisahkan oleh citra dengan jelas.
- **Mikroskop bukan penyebab cluster**: proporsi M2 per cluster 0,388–0,407, dan hanya 0,54% sampel yang ambigu
  menurut GMM (posterior cluster kedua > 0,2).

**Keterbatasan**: bukti bahwa cluster 1 dan 6 adalah dua material adalah yang paling lemah di antara ketiga
belahan. Jika keduanya ternyata satu material, k yang tepat adalah 7. ARI terhadap label sebenarnya hanya diketahui
dosen.

## Referensi
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020). A Simple Framework for Contrastive Learning of
  Visual Representations. ICML.
- He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Xie, J., Girshick, R., Farhadi, A. (2016). Unsupervised Deep Embedding for Clustering Analysis. ICML.
- McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension
  Reduction. arXiv:1802.03426.
- Radford, A. dkk. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML.
