# Tabel angka laporan (dibuat otomatis oleh `kode/tabel_laporan.py`)

Setiap angka berasal dari `runs/<run>/metrics.json` atau CSV di `runs/*_tabel/`.

## Task 1 — Klasifikasi multimodal

**Sanity check overfit 1 batch (64 sampel):** loss akhir 0.0001, akurasi akhir 1.0000 setelah 300 langkah.

### Modality & fusion (validasi 9.600 unit)

| run | macro_f1 | macro_f1_lini_A | macro_f1_lini_B | macro_f1_bobot_test | macro_f1_tanpa_sensor | waktu_s |
|---|---|---|---|---|---|---|
| t1_mod_image | 0.6872 | 0.7210 | 0.5835 | 0.6394 | 0.6872 | 328.7000 |
| t1_mod_sensor | 0.4579 | 0.4709 | 0.4169 | 0.4392 | 0.0949 | 58.6000 |
| t1_mod_concat | 0.8371 | 0.8700 | 0.7356 | 0.7903 | 0.6415 | 594.3000 |
| t1_mod_late | 0.8298 | 0.8619 | 0.7295 | 0.7838 | 0.6221 | 624.5000 |
| t1_mod_xattn | 0.8440 | 0.8731 | 0.7539 | 0.8025 | 0.6164 | 459.6000 |

### Label noise

| run | macro_f1 | macro_f1_lini_A | macro_f1_lini_B | macro_f1_bobot_test | macro_f1_tanpa_sensor | waktu_s |
|---|---|---|---|---|---|---|
| t1_mod_xattn | 0.8440 | 0.8731 | 0.7539 | 0.8025 | 0.6164 | 459.6000 |
| t1_noise_ls_xattn | 0.8436 | 0.8726 | 0.7561 | 0.8030 | 0.6261 | 956.4000 |
| t1_noise_buang_xattn | 0.8521 | 0.8827 | 0.7593 | 0.8091 | 0.6235 | 1058.9000 |

### Domain shift

| run | macro_f1 | macro_f1_lini_A | macro_f1_lini_B | macro_f1_bobot_test | macro_f1_tanpa_sensor | waktu_s |
|---|---|---|---|---|---|---|
| t1_mod_xattn | 0.8440 | 0.8731 | 0.7539 | 0.8025 | 0.6164 | 459.6000 |
| t1_dom_perline_xattn | 0.8402 | 0.8720 | 0.7439 | 0.7955 | 0.6202 | 915.9000 |
| t1_dom_aug_xattn | 0.8462 | 0.8798 | 0.7439 | 0.7988 | 0.6153 | 1432.2000 |
| t1_dom_reweight_xattn | 0.8380 | 0.8674 | 0.7490 | 0.7967 | 0.5664 | 507.0000 |
| t1_dom_perline_aug_xattn | 0.8371 | 0.8717 | 0.7329 | 0.7886 | 0.6048 | 501.0000 |
| t1_dom_tekstur_xattn | 0.8379 | 0.8677 | 0.7469 | 0.7958 | 0.5772 | 742.0000 |

### Missing modality

| run | macro_f1 | macro_f1_lini_A | macro_f1_lini_B | macro_f1_bobot_test | macro_f1_tanpa_sensor | waktu_s | macro_f1_sensor_hilang_asli |
|---|---|---|---|---|---|---|---|
| t1_mod_xattn | 0.8440 | 0.8731 | 0.7539 | 0.8025 | 0.6164 | 459.6000 | 0.6217 |
| t1_miss_drop25_xattn | 0.8692 | 0.8969 | 0.7824 | 0.8295 | 0.6705 | 585.5000 | 0.6841 |

### F1 per kelas per modality

| run | f1_normal | f1_gores | f1_noda | f1_retak | f1_gelembung | f1_las_dingin | f1_kontaminasi |
|---|---|---|---|---|---|---|---|
| t1_mod_image | 0.8541 | 0.7686 | 0.8813 | 0.6496 | 0.7155 | 0.0000 | 0.9414 |
| t1_mod_sensor | 0.6528 | 0.1075 | 0.0970 | 0.6078 | 0.7365 | 0.9292 | 0.0744 |
| t1_mod_concat | 0.9074 | 0.7607 | 0.8656 | 0.6692 | 0.7845 | 0.9313 | 0.9409 |
| t1_mod_late | 0.9029 | 0.7448 | 0.8665 | 0.6667 | 0.7674 | 0.9257 | 0.9343 |
| t1_mod_xattn | 0.9076 | 0.7732 | 0.8773 | 0.6894 | 0.7905 | 0.9364 | 0.9335 |

**Kandidat salah label (OOF 5-fold):** 1241 unit (2.59%); macro-F1 OOF 0.8647. Pasangan terbanyak: {'normal->las_dingin': 161, 'retak->normal': 131, 'gores->normal': 121, 'gelembung->noda': 121, 'noda->gelembung': 107, 'retak->gores': 104, 'gores->retak': 95, 'las_dingin->normal': 92}.

### Model final (3 seed, mean ± std)

| metrik | mean | std |
|---|---|---|
| macro_f1 | 0.8700 | 0.0012 |
| macro_f1_lini_A | 0.8992 | 0.0022 |
| macro_f1_lini_B | 0.7798 | 0.0031 |
| macro_f1_bobot_test | 0.8286 | 0.0015 |
| macro_f1_tanpa_sensor | 0.6704 | 0.0023 |
| f1_normal | 0.9220 | 0.0011 |
| f1_gores | 0.8106 | 0.0088 |
| f1_noda | 0.8856 | 0.0033 |
| f1_retak | 0.7492 | 0.0036 |
| f1_gelembung | 0.8374 | 0.0023 |
| f1_las_dingin | 0.9412 | 0.0004 |
| f1_kontaminasi | 0.9444 | 0.0053 |

## Task 2 — RUL

Sensor dibuang: s01, s05, s06, s10, s16, s18, s19. Spike diganti 5199, NaN diisi 72200 (train).

### Perbandingan model (5-fold per mesin, 5 titik potong acak per mesin validasi)

| run | rmse | skor_total | skor_per_titik | rmse_fold_mean | rmse_fold_std | rmse_rul_0_50 | skor_rul_0_50 | rmse_rul_50_100 | skor_rul_50_100 | rmse_rul_100_150 | skor_rul_100_150 | rmse_semua_titik |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2_cv_lgbm | 20.1668 | 121527.2986 | 27.0061 | 20.1558 | 0.7441 | 12.8912 | 11694.2524 | 30.9945 | 98336.8337 | 16.4352 | 11496.2125 | 17.2395 |
| t2_cv_lgbm_W50 | 20.1297 | 115158.2269 | 25.5907 | 20.1194 | 0.7198 | 12.6362 | 6582.3758 | 30.7652 | 96956.8241 | 16.6287 | 11619.0270 | 17.1768 |
| t2_cv_gru_awal | 24.1929 | 271341.9438 | 60.2982 | 24.1787 | 0.9275 | 15.1148 | 41329.8744 | 34.4259 | 203645.7913 | 21.9348 | 26366.2782 | 21.1769 |
| t2_cv_gru | 19.5358 | 97209.7077 | 21.6022 | 19.5255 | 0.7094 | 11.9519 | 5388.3611 | 29.5013 | 80394.2683 | 16.5300 | 11427.0784 | 16.8179 |
| t2_cv_tcn | 20.6547 | 102579.3989 | 22.7954 | 20.6246 | 1.2461 | 19.1000 | 14102.0565 | 29.3183 | 78292.4171 | 16.2061 | 10184.9253 | 17.7854 |
| t2_cv_transformer | 19.8046 | 119409.4887 | 26.5354 | 19.8002 | 0.4674 | 11.1490 | 4860.4767 | 29.5005 | 100244.3875 | 17.3705 | 14304.6245 | 17.2047 |
| t2_cv_gru_W30 | 19.8949 | 103598.8290 | 23.0220 | 19.8837 | 0.7461 | 12.6984 | 7862.4732 | 30.2010 | 84457.2422 | 16.5297 | 11279.1136 | 17.1527 |

### Ketidakpastian (t2_uncert_gru, fold 0, 900 titik)

| metrik | nilai |
|---|---|
| rmse_seed0 | 20.0490 |
| rmse_ensemble | 19.7293 |
| skor_seed0 | 18929.6411 |
| skor_ensemble | 18609.0629 |
| cakupan_mentah_mc_dropout_seed0 | 0.5722 |
| lebar_mentah_mc_dropout_seed0 | 24.3422 |
| cakupan_mentah_ensemble_saja | 0.1644 |
| cakupan_mentah_ensemble+mc | 0.6222 |
| lebar_mentah_ensemble+mc | 27.2623 |
| cakupan_conformal_split | 0.9033 |
| lebar_conformal_split | 56.1022 |
| faktor_conformal_q | 3.9107 |
| korelasi_sd_vs_galat_abs | 0.0227 |

Cakupan per rentang: {'0-50': 0.915, '50-100': 0.723, '100-150': 0.977}

| T | aturan | tertangkap_rusak<=10 | alarm_palsu_rul>50 | frac_diservis | rata2_sisa_umur_terbuang |
|---|---|---|---|---|---|
| 10 | prediksi_titik | 0.3333 | 0.0000 | 0.0200 | 6.7778 |
| 10 | batas_bawah_90 | 1.0000 | 0.0057 | 0.1578 | 19.4437 |
| 20 | prediksi_titik | 0.6923 | 0.0000 | 0.0689 | 13.0484 |
| 20 | batas_bawah_90 | 1.0000 | 0.0128 | 0.1800 | 22.1481 |
| 30 | prediksi_titik | 1.0000 | 0.0029 | 0.1333 | 16.8750 |
| 30 | batas_bawah_90 | 1.0000 | 0.0185 | 0.2122 | 24.6335 |
| 40 | prediksi_titik | 1.0000 | 0.0071 | 0.1700 | 20.7386 |
| 40 | batas_bawah_90 | 1.0000 | 0.0514 | 0.2456 | 29.9774 |

Geser bias: {'c_terbaik_skor': -15, 'skor_c0': 18609.062884928597, 'skor_terbaik': 7942.289908074972, 'rmse_c0': 19.729291026765363, 'rmse_di_c_terbaik': 22.572301775987977}

**Prediksi final:** 300 mesin, RUL rata-rata 94.08, 24.0% direkomendasikan servis sekarang (batas bawah 90% ≤ 30).

## Task 3 — Clustering

### Perbandingan representasi (tanpa label)

| k | img | spec | ARI_img_vs_spec | NMI_img_mikroskop | NMI_spec_mikroskop | sil_img | sil_spec |
|---|---|---|---|---|---|---|---|
| 6 | simclr_img | simclr_spec | 0.5955 | 0.0000 | 0.0000 | 0.4424 | 0.2754 |
| 6 | simclr_img | joint_spec | 0.5955 | 0.0000 | 0.0000 | 0.4424 | 0.2697 |
| 6 | simclr_img | klasik_spec (median+baseline+PCA) | 0.6407 | 0.0000 | 0.0001 | 0.4424 | 0.2591 |
| 6 | joint_img | simclr_spec | 0.5957 | 0.0000 | 0.0000 | 0.4145 | 0.2754 |
| 6 | joint_img | joint_spec | 0.5957 | 0.0000 | 0.0000 | 0.4145 | 0.2697 |
| 6 | joint_img | klasik_spec (median+baseline+PCA) | 0.6410 | 0.0000 | 0.0001 | 0.4145 | 0.2591 |
| 6 | pretrained_resnet18 | simclr_spec | 0.4701 | 0.0066 | 0.0000 | 0.3683 | 0.2754 |
| 6 | pretrained_resnet18 | joint_spec | 0.4700 | 0.0066 | 0.0000 | 0.3683 | 0.2697 |
| 6 | pretrained_resnet18 | klasik_spec (median+baseline+PCA) | 0.5235 | 0.0066 | 0.0001 | 0.3683 | 0.2591 |
| 10 | simclr_img | simclr_spec | 0.3636 | 0.0076 | 0.0001 | 0.2878 | 0.2459 |
| 10 | simclr_img | joint_spec | 0.3639 | 0.0076 | 0.0001 | 0.2878 | 0.2370 |
| 10 | simclr_img | klasik_spec (median+baseline+PCA) | 0.3125 | 0.0076 | 0.0001 | 0.2878 | 0.2420 |
| 10 | joint_img | simclr_spec | 0.3559 | 0.0067 | 0.0001 | 0.2431 | 0.2459 |
| 10 | joint_img | joint_spec | 0.3570 | 0.0067 | 0.0001 | 0.2431 | 0.2370 |
| 10 | joint_img | klasik_spec (median+baseline+PCA) | 0.3224 | 0.0067 | 0.0001 | 0.2431 | 0.2420 |
| 10 | pretrained_resnet18 | simclr_spec | 0.3359 | 0.0138 | 0.0001 | 0.3283 | 0.2459 |
| 10 | pretrained_resnet18 | joint_spec | 0.3362 | 0.0138 | 0.0001 | 0.3283 | 0.2370 |
| 10 | pretrained_resnet18 | klasik_spec (median+baseline+PCA) | 0.2933 | 0.0138 | 0.0001 | 0.3283 | 0.2420 |

### Pemilihan jumlah cluster

Pilihan per kriteria: {'k_silhouette': 5, 'k_davies_bouldin': 5, 'k_bic': 20, 'k_bic_siku': 7, 'k_silhouette_umap': 7, 'n_komponen_knn_umap': {'10': 7, '30': 7}, 'k_bertingkat': 8, 'uji_belah': [{'cluster_kasar': 0, 'n': 10043, 'lembah_citra': 1.0, 'lembah_spektrum': 0.0039, 'lembah_fusion': 0.0034, 'terbelah': True}, {'cluster_kasar': 1, 'n': 8685, 'lembah_citra': 1.0, 'lembah_spektrum': 0.2019, 'lembah_fusion': 0.1994, 'terbelah': True}, {'cluster_kasar': 2, 'n': 3656, 'lembah_citra': 0.9723, 'lembah_spektrum': 0.4404, 'lembah_fusion': 0.4553, 'terbelah': False}, {'cluster_kasar': 3, 'n': 4979, 'lembah_citra': 1.0, 'lembah_spektrum': 0.416, 'lembah_fusion': 0.4145, 'terbelah': False}, {'cluster_kasar': 4, 'n': 8637, 'lembah_citra': 0.0033, 'lembah_spektrum': 0.5019, 'lembah_fusion': 0.0033, 'terbelah': True}]}

| k | silhouette | davies_bouldin | bic_gmm | stabilitas_ari | inersia | silhouette_umap | ARI_citra_vs_spektrum |
|---|---|---|---|---|---|---|---|
| 2.0000 | 0.1947 | 2.0431 | -713919.6778 | 1.0000 | 28963.9485 | 0.4144 | 0.5165 |
| 3.0000 | 0.2399 | 1.7036 | -894649.0498 | 0.8011 | 24116.8879 | 0.5338 | 0.4123 |
| 4.0000 | 0.2995 | 1.4606 | -1061177.4162 | 1.0000 | 19488.8665 | 0.6110 | 0.8007 |
| 5.0000 | 0.3155 | 1.3319 | -1104471.2649 | 0.9343 | 17487.4858 | 0.6544 | 0.8344 |
| 6.0000 | 0.3076 | 1.3767 | -1135631.1521 | 0.8395 | 16103.7218 | 0.7449 | 0.5955 |
| 7.0000 | 0.2574 | 1.6593 | -1159397.3738 | 0.7768 | 15520.0244 | 0.8442 | 0.5366 |
| 8.0000 | 0.2293 | 1.7824 | -1158239.4956 | 0.8940 | 14608.1428 | 0.7872 | 0.4315 |
| 9.0000 | 0.2060 | 1.8919 | -1191361.4183 | 0.9147 | 14102.1358 | 0.7377 | 0.3818 |
| 10.0000 | 0.1998 | 1.9042 | -1200958.2849 | 0.9353 | 13776.2201 | 0.6513 | 0.3639 |
| 11.0000 | 0.1850 | 1.9647 | -1210761.7261 | 0.8162 | 13419.9890 | 0.6010 | 0.3152 |
| 12.0000 | 0.1794 | 2.0292 | -1222952.0855 | 0.8682 | 13097.3579 | 0.5565 | 0.2878 |
| 13.0000 | 0.1693 | 2.0640 | -1228720.6022 | 0.8380 | 12811.5196 | 0.5116 | 0.2581 |
| 14.0000 | 0.1556 | 2.1187 | -1242115.0496 | 0.8067 | 12548.0386 | 0.4928 | 0.2378 |
| 15.0000 | 0.1550 | 2.1385 | -1245771.1043 | 0.8297 | 12415.2098 | 0.4466 | 0.2263 |
| 16.0000 | 0.1513 | 2.1834 | -1253489.8239 | 0.9093 | 12038.9383 | 0.4403 | 0.2194 |
| 17.0000 | 0.1508 | 2.1347 | -1248117.5813 | 0.8802 | 11914.5789 | 0.4230 | 0.1904 |
| 18.0000 | 0.1460 | 2.1346 | -1261445.3435 | 0.8178 | 11769.7313 | 0.4104 | 0.1832 |
| 19.0000 | 0.1461 | 2.1099 | -1262368.5455 | 0.7994 | 11566.3951 | 0.4162 | 0.1737 |
| 20.0000 | 0.1413 | 2.1042 | -1265473.1524 | 0.8709 | 11380.3855 | 0.4025 | 0.1630 |

### Final (k = 8, metode kmeans)

| metode | silhouette | DB | NMI mikroskop | stabilitas 3 seed |
|---|---|---|---|---|
| kmeans | 0.2293 | 1.7824 | 0.0000 | 1.0000 |
| gmm | 0.2239 | 1.8152 | 0.0000 | 0.8013 |
| dec | 0.2292 | 1.7838 | 0.0001 | 0.8835 |

ARI antar metode: {'kmeans_vs_gmm': 0.8164246580611302, 'kmeans_vs_dec': 0.9910016588890842, 'gmm_vs_dec': 0.8137304307572967}

Modality: {'image_only': {'silhouette': 0.2997821569442749, 'ARI_vs_fusion': 0.6925566212333345, 'nmi_mikroskop': 0.007643257314991278}, 'spectrum_only': {'silhouette': 0.25240838527679443, 'ARI_vs_fusion': 0.7397835617431083, 'nmi_mikroskop': 4.700101417556039e-05}, 'ARI_image_vs_spectrum': 0.43153145804465626}

Cluster sulit: {'silhouette_per_cluster': {'0': 0.2425, '1': 0.203, '2': 0.1635, '3': 0.3578, '4': 0.2622, '5': 0.3177, '6': 0.1836, '7': 0.148}, 'pasangan_terdekat': [{'pasangan': [2, 7], 'jarak_pusat': 0.5380366282611444}, {'pasangan': [1, 6], 'jarak_pusat': 0.6483881986212285}, {'pasangan': [0, 4], 'jarak_pusat': 0.8005464574904836}, {'pasangan': [3, 5], 'jarak_pusat': 0.9743654322095868}], 'frac_sampel_ambigu_gmm(p2>0.2)': 0.005388888888888889, 'mikroskop_per_cluster_frac_M2': {'0': 0.399, '1': 0.388, '2': 0.4, '3': 0.407, '4': 0.4, '5': 0.402, '6': 0.395, '7': 0.4}}

## Daftar figur

- `laporan_figs\task1\domain_fourier.png`
- `laporan_figs\task1\eda_contoh_gambar.png`
- `laporan_figs\task1\eda_distribusi.png`
- `laporan_figs\task1\eda_shift_lini.png`
- `laporan_figs\task1\eda_sinyal_per_kelas.png`
- `laporan_figs\task1\final_confusion.png`
- `laporan_figs\task1\final_kurva.png`
- `laporan_figs\task1\modality_f1_heatmap.png`
- `laporan_figs\task1\noise_contoh.png`
- `laporan_figs\task1\noise_matriks.png`
- `laporan_figs\task2\eda_contoh_mesin.png`
- `laporan_figs\task2\eda_panjang.png`
- `laporan_figs\task2\eda_regime.png`
- `laporan_figs\task2\eda_sensor.png`
- `laporan_figs\task2\uncert_kalibrasi.png`
- `laporan_figs\task2\uncert_lintasan.png`
- `laporan_figs\task3\contoh_cluster.png`
- `laporan_figs\task3\eda_contoh.png`
- `laporan_figs\task3\eda_distorsi.png`
- `laporan_figs\task3\modality_kontingensi.png`
- `laporan_figs\task3\pilih_k.png`
- `laporan_figs\task3\umap.png`
