# Task 3 — clustering multimodal tanpa label (gambar mikrostruktur + spektrum)

Model utama: encoder SimCLR (PyTorch) untuk gambar dan spektrum, fusi embedding, lalu clustering.
Tidak ada label yang dipakai untuk training. Konfigurasi final (representasi, k, metode) ada di
`FINAL` pada `kode/reproduce.py`.

| File | Isi |
|---|---|
| `task3_eda.py` | 3a — distorsi pengganggu tanpa label: kecerahan per mikroskop, rotasi/skala gambar, pergeseran puncak, baseline, spike spektrum. |
| `task3_ssl.py` | 3b–c — SimCLR gambar, SimCLR spektrum, SimCLR gabungan (`--modal {img,spec,joint}`), dan ResNet18 ImageNet sebagai pembanding (`--modal pretrained`). |
| `task3_cluster.py` | 3c–f — fusi embedding, `--tahap banding` (pilih representasi), `k` (pilih jumlah cluster), `final` (K-Means vs GMM vs DEC + analisis). |

Urutan: `task3_eda.py` → `task3_ssl.py` (4 varian) → `task3_cluster.py --tahap banding` → `k` → `final`.

```bash
python kode/task3_cluster/task3_ssl.py --data data_kecil --modal img --smoke  # uji cepat CPU
python kode/reproduce.py --data data_uts_dl --task 3                          # angka laporan dari weights/
```

Output: `runs/t3_*/`, `laporan_figs/task3/`, `cluster.csv`.

Pretrained: ResNet18 bobot ImageNet dari torchvision (He dkk., 2016), hanya sebagai pembanding.
