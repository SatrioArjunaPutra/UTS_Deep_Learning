# Task 2 — prediksi Remaining Useful Life (RUL)

Dikerjakan oleh Marchel (PR Task 2). Folder ini menampung skrip Task 2:

| File | Isi |
|---|---|
| `task2_eda.py` | 2a — eksplorasi: regime operasi, sensor tak informatif, NaN, spike, panjang riwayat. |
| `task2_prep.py` | 2a–c — preprocessing (normalisasi per regime, statistik dari train saja), label RUL, sliding window kausal. |
| `task2_model.py` | 2d — tiga arsitektur sekuens (GRU, TCN, Transformer), semuanya punya dropout untuk MC dropout. |
| `task2_train.py` | 2d–e — training & validasi; GroupKFold 5 fold per mesin, baseline LightGBM. |
| `task2_eksperimen.py` | Grup eksperimen `--grup {tune,model,window,uncert,final}`. |

```bash
python kode/reproduce.py --data data_uts_dl --task 2                         # angka laporan dari weights/
```

Output: `runs/t2_*/`, `laporan_figs/task2/`, `prediksi_rul.csv`.
