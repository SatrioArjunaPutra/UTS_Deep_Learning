# common — utilitas bersama

Dipakai oleh semua task. Isi folder ini tidak melatih model.

| File | Isi |
|---|---|
| `__init__.py` | Paket `common`: `set_seed`, `get_device`, `buat_run` (folder `runs/<nama>/`), `simpan_json`, `simpan_metrics`, `versi_library`, `folder_figur`. Setiap run menyimpan `config.json` (argumen + versi library + seed) dan `metrics.json` (sumber angka laporan). |
| `buat_subset.py` | Membuat `data_kecil/` (subset dengan struktur file sama) untuk uji cepat di CPU. |
| `ringkas_data.py` | Meringkas isi folder data menjadi `docs/ringkasan_data.txt` (dipakai sekali, sebelum kode task ditulis). |

`metrik.py` (macro-F1, RMSE, skor asimetris, dll.) sengaja tetap di `kode/` karena juga dipakai
`cek_submission.py` dan skrip Task 2 dengan `from metrik import ...`.

```bash
python kode/common/buat_subset.py --data data_uts_dl --out data_kecil
```

Cara skrip task menemukan paket ini: tiap skrip task menambahkan folder task-nya sendiri dan
`kode/` ke `sys.path`, lalu `from common import ...`.
