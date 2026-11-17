"""Task 3c-f — fusion embedding, pemilihan jumlah cluster, K-Means vs GMM vs DEC, analisis.

Langkah:
1. Ruang embedding per modality: standardisasi -> PCA (32 dim) -> normalisasi L2.
   Fusion = concat [sqrt(w)*img, sqrt(1-w)*spec] (tiap blok norma 1 sehingga bobotnya jelas).
2. --tahap banding: membandingkan kandidat representasi TANPA label. Kriteria utama =
   kesepakatan lintas modality: ARI antara cluster dari gambar saja dan cluster dari spektrum
   saja. Jenis material memengaruhi kedua modality, sedangkan pengganggu (mikroskop, rotasi,
   baseline) hanya memengaruhi salah satunya; jadi representasi yang lebih menangkap material
   menghasilkan kesepakatan lebih tinggi. Juga dicatat NMI cluster vs mikroskop (harus rendah).
3. --tahap k: scan k=2..20 pada ruang fusion: silhouette (maks), Davies-Bouldin (min), BIC GMM
   (min / siku), stabilitas = rata-rata ARI antar 5 seed K-Means, kesepakatan citra-vs-spektrum
   per k, dan dua kriteria di ruang UMAP 2-D fusion: silhouette (maks) dan jumlah komponen
   terhubung graf kNN. Alasan kriteria UMAP: beberapa material hanya berbeda di SATU modality;
   setelah concat perbedaan itu "terencerkan" setengahnya, sehingga silhouette/DB di ruang 64-d
   (yang menyukai cluster bulat) menggabungkannya. UMAP menjaga struktur lokal sehingga pulau-pulau
   itu tetap terpisah. Kriteria terakhir = uji belah bertingkat: mulai dari k kasar (silhouette
   64-d), tiap cluster diproyeksikan ke arah 2-means di ruang CITRA dan SPEKTRUM terpisah; jika
   histogram proyeksinya punya lembah < AMBANG_LEMBAH x puncak terendah (dua puncak), cluster itu
   dihitung dua material. k_bertingkat = k_kasar + jumlah cluster yang terbelah (laporan 3e).
4. --tahap final: pada k terpilih bandingkan K-Means, GMM, DEC (Xie dkk., 2016; autoencoder +
   inisialisasi K-Means + penyempurnaan KL ke distribusi target), image-only vs spectrum-only vs
   fusion, UMAP, cluster sulit dipisah, dan tulis cluster.csv. --metode auto memilih metode dengan
   silhouette fusion tertinggi (dibulatkan 3 desimal), seri -> stabilitas antar seed tertinggi.

  python kode/task3_cluster/task3_cluster.py --tahap banding
  python kode/task3_cluster/task3_cluster.py --tahap k   --img runs/t3_ssl_joint_s0/emb_img.npy --spec runs/t3_ssl_joint_s0/emb_spec.npy
  python kode/task3_cluster/task3_cluster.py --tahap final --k 8 --metode dec ...
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,  # noqa: E402
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# folder task ini (modul task) + kode/ (paket common) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import ROOT, buat_run, folder_figur, set_seed, simpan_json, simpan_metrics  # noqa: E402

RUNS = os.path.join(ROOT, "runs")


# ------------------------------------------------------------------ ruang embedding
def ruang(e, dim=32, seed=0):
    x = StandardScaler().fit_transform(e)
    x = PCA(min(dim, x.shape[1]), random_state=seed).fit_transform(x)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fusi(xi, xs, w=0.5):
    return np.concatenate([np.sqrt(w) * xi, np.sqrt(1 - w) * xs], 1)


def spektrum_klasik(sp):
    """Pembanding non-DL: median filter (hapus spike) - baseline linear (persentil bawah)."""
    from scipy.ndimage import median_filter

    mf = median_filter(sp.astype(np.float32), size=(1, 5))
    t = np.arange(sp.shape[1])
    out = np.empty_like(mf)
    for i, x in enumerate(mf):
        m = x <= np.percentile(x, 30)
        out[i] = x - np.polyval(np.polyfit(t[m], x[m], 1), t)
    return out


def kmeans(x, k, seed=0, n_init=10):
    return KMeans(k, n_init=n_init, random_state=seed).fit_predict(x)


def sil(x, lab, n=8000, seed=0):
    return float(silhouette_score(x, lab, sample_size=min(n, len(x)), random_state=seed))


# ------------------------------------------------------------------ DEC
def dec(x, k, seed=0, dev=None, z_dim=16, pre_ep=60, dec_iter=40, gamma=0.1, log=print):
    """Deep Embedded Clustering (varian IDEC: KL + gamma * rekonstruksi agar struktur lokal terjaga).

    1) autoencoder MLP d->256->128->z dilatih rekonstruksi; 2) K-Means di ruang z memberi pusat
    awal; 3) soft assignment Student-t q_ij, target p_ij = q^2/f_j (dinormalisasi) yang
    mempertajam assignment yakin; minimalkan KL(P||Q). P diperbarui tiap iterasi (epoch).
    """
    import torch
    import torch.nn as nn

    set_seed(seed)
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(x.astype(np.float32)).to(dev)
    d = X.shape[1]
    enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, z_dim)).to(dev)
    decd = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, d)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(decd.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    n, bs = len(X), 512
    for ep in range(pre_ep):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            b = X[perm[i:i + bs]]
            loss = ((decd(enc(b)) - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = enc(X).cpu().numpy()
    km = KMeans(k, n_init=20, random_state=seed).fit(Z)
    mu = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(list(enc.parameters()) + list(decd.parameters()) + [mu], lr=1e-4)

    def q_of(z):
        q = 1.0 / (1.0 + torch.cdist(z, mu) ** 2)
        return q / q.sum(1, keepdim=True)

    lab_lama = km.labels_
    perubahan = []
    for it in range(dec_iter):
        with torch.no_grad():
            q = q_of(enc(X))
            p = q ** 2 / q.sum(0)
            p = p / p.sum(1, keepdim=True)
            lab = q.argmax(1).cpu().numpy()
        ubah = float((lab != lab_lama).mean())
        perubahan.append(ubah)
        lab_lama = lab
        if it > 0 and ubah < 0.001:
            break
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            z = enc(X[idx])
            kl = (p[idx] * (torch.log(p[idx] + 1e-10) - torch.log(q_of(z) + 1e-10))).sum(1).mean()
            loss = kl + gamma * ((decd(z) - X[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = enc(X)
        lab = q_of(Z).argmax(1).cpu().numpy()
    log(f"   DEC: {it + 1} iterasi, perubahan label terakhir {perubahan[-1]:.4f}, ukuran cluster {np.bincount(lab, minlength=k).tolist()}")
    return lab, Z.cpu().numpy(), {"iterasi": it + 1, "perubahan_label": perubahan, "init_kmeans": km.labels_}


# ------------------------------------------------------------------ tahap
def muat_emb(p):
    return np.load(p if os.path.isabs(p) else os.path.join(ROOT, p))


def tahap_banding(a, mik, sp):
    """Bandingkan kandidat representasi gambar & spektrum tanpa label."""
    kand_img = {k: v for k, v in {
        "simclr_img": "runs/t3_ssl_img_s0/emb_img.npy",
        "joint_img": "runs/t3_ssl_joint_s0/emb_img.npy",
        "pretrained_resnet18": "runs/t3_ssl_pretrained_s0/emb_img.npy"}.items() if os.path.exists(os.path.join(ROOT, v))}
    kand_spec = {k: v for k, v in {
        "simclr_spec": "runs/t3_ssl_spec_s0/emb_spec.npy",
        "joint_spec": "runs/t3_ssl_joint_s0/emb_spec.npy"}.items() if os.path.exists(os.path.join(ROOT, v))}
    R_img = {k: ruang(muat_emb(v)) for k, v in kand_img.items()}
    R_spec = {k: ruang(muat_emb(v)) for k, v in kand_spec.items()}
    R_spec["klasik_spec (median+baseline+PCA)"] = ruang(spektrum_klasik(sp))
    rows = []
    for k in a.k_banding:
        lab_i = {n: kmeans(x, k) for n, x in R_img.items()}
        lab_s = {n: kmeans(x, k) for n, x in R_spec.items()}
        for ni, li in lab_i.items():
            for ns, ls in lab_s.items():
                rows.append({"k": k, "img": ni, "spec": ns, "ARI_img_vs_spec": adjusted_rand_score(li, ls),
                             "NMI_img_mikroskop": normalized_mutual_info_score(li, mik),
                             "NMI_spec_mikroskop": normalized_mutual_info_score(ls, mik),
                             "sil_img": sil(R_img[ni], li), "sil_spec": sil(R_spec[ns], ls)})
                print(rows[-1])
    df = pd.DataFrame(rows)
    return df


AMBANG_LEMBAH = 0.3   # cluster satu gumpalan memberi rasio ~0,4-0,5 (diamati), dua puncak terpisah ~0


def rasio_lembah(z, bins=60):
    """Rasio titik terendah histogram di antara dua puncak (kiri & kanan) terhadap puncak terendah.
    ~0 = dua puncak terpisah jelas, ~1 = satu gumpalan. Histogram dihaluskan rata-rata bergerak 5 bin."""
    h, _ = np.histogram(z, bins=bins)
    h = np.convolve(h, np.ones(5) / 5, "same")
    a, b = int(np.argmax(h[:bins // 2])), bins // 2 + int(np.argmax(h[bins // 2:]))
    return float(h[a:b + 1].min() / max(min(h[a], h[b]), 1e-9))


def uji_belah(X, Xi, Xs, k_kasar):
    """Untuk tiap cluster K-Means k_kasar: rasio lembah proyeksi 2-means di citra, spektrum, fusion."""
    lab = kmeans(X, k_kasar)
    rows = []
    for c in range(k_kasar):
        m = lab == c
        r = {"cluster_kasar": c, "n": int(m.sum())}
        for nm, Z in [("citra", Xi), ("spektrum", Xs), ("fusion", X)]:
            l2 = kmeans(Z[m], 2)
            d = Z[m][l2 == 1].mean(0) - Z[m][l2 == 0].mean(0)
            r[f"lembah_{nm}"] = rasio_lembah(Z[m] @ d)
        r["terbelah"] = bool(min(r["lembah_citra"], r["lembah_spektrum"]) < AMBANG_LEMBAH)
        rows.append(r)
    return pd.DataFrame(rows)


def komponen_knn(Z, nn=10):
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    return connected_components(kneighbors_graph(Z, nn, include_self=False), directed=False)


def tahap_k(a, X, run, Xi=None, Xs=None):
    rows = []
    t0 = time.time()
    U = umap2d(X)
    np.save(os.path.join(run, "umap_fusion.npy"), U)
    n_komp = {nn: int(komponen_knn(U, nn)[0]) for nn in (10, 30)}
    print(f"UMAP {time.time() - t0:.0f}s; komponen kNN di UMAP: {n_komp}")
    for k in range(a.k_min, a.k_max + 1):
        labs = [kmeans(X, k, seed=s, n_init=4) for s in range(5)]
        stab = np.mean([adjusted_rand_score(labs[i], labs[j]) for i in range(5) for j in range(i + 1, 5)])
        lab = kmeans(X, k)
        Xg = PCA(16, random_state=0).fit_transform(X)
        gmm = GaussianMixture(k, covariance_type="full", n_init=2, random_state=0).fit(Xg)
        rows.append({"k": k, "silhouette": sil(X, lab), "davies_bouldin": float(davies_bouldin_score(X, lab)),
                     "bic_gmm": float(gmm.bic(Xg)), "stabilitas_ari": float(stab),
                     "inersia": float(KMeans(k, n_init=4, random_state=0).fit(X).inertia_),
                     "silhouette_umap": sil(U, lab),
                     "ARI_citra_vs_spektrum": float(adjusted_rand_score(kmeans(Xi, k), kmeans(Xs, k))) if Xi is not None else None})
        print(f"k={k:2d} sil={rows[-1]['silhouette']:.4f} DB={rows[-1]['davies_bouldin']:.3f} "
              f"BIC={rows[-1]['bic_gmm']:.0f} stab={stab:.3f} sil_umap={rows[-1]['silhouette_umap']:.4f} "
              f"ARI_i_s={rows[-1]['ARI_citra_vs_spektrum']:.3f} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run, "scan_k.csv"), index=False)
    fig, ax = plt.subplots(2, 3, figsize=(16, 7))
    ax = ax.ravel()
    for axi, c, judul in zip(ax, ["silhouette", "davies_bouldin", "bic_gmm", "stabilitas_ari", "silhouette_umap",
                                  "ARI_citra_vs_spektrum"],
                             ["Silhouette ruang fusion 64-d (maks)", "Davies-Bouldin 64-d (min)", "BIC GMM (min/siku)",
                              "Stabilitas antar seed (ARI)", "Silhouette ruang UMAP 2-D (maks)",
                              "Kesepakatan citra vs spektrum (ARI)"]):
        axi.plot(df.k, df[c], "o-")
        axi.set_title(judul); axi.set_xlabel("k"); axi.set_xticks(df.k[::2])
    fig.tight_layout(); fig.savefig(os.path.join(folder_figur(3), "pilih_k.png"), dpi=120); plt.close(fig)
    pilih = {"k_silhouette": int(df.k[df.silhouette.idxmax()]), "k_davies_bouldin": int(df.k[df.davies_bouldin.idxmin()]),
             "k_bic": int(df.k[df.bic_gmm.idxmin()])}
    # siku BIC: k terkecil setelah itu penurunan BIC < 1% dari total rentang
    d = -np.diff(df.bic_gmm.values)
    rng = df.bic_gmm.max() - df.bic_gmm.min()
    siku = next((int(df.k[i]) for i in range(len(d)) if d[i] < 0.01 * rng), int(df.k.iloc[-1]))
    pilih["k_bic_siku"] = siku
    pilih["k_silhouette_umap"] = int(df.k[df.silhouette_umap.idxmax()])
    pilih["n_komponen_knn_umap"] = n_komp
    if Xi is not None:
        ub = uji_belah(X, Xi, Xs, pilih["k_silhouette"])
        ub.round(3).to_csv(os.path.join(run, "uji_belah.csv"), index=False)
        print(ub.round(3).to_string(index=False))
        pilih["k_bertingkat"] = int(pilih["k_silhouette"] + ub.terbelah.sum())
        pilih["uji_belah"] = ub.round(4).to_dict(orient="records")
    return df, pilih


def umap2d(X, seed=0):
    import umap

    return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(X)


def tahap_final(a, Xi, Xs, X, mik, g8, sp, run):
    k = a.k
    hasil = {"k": k}
    labs = {}
    t0 = time.time()
    sd = a.seed
    labs["kmeans"] = kmeans(X, k, seed=sd)
    Xg = PCA(16, random_state=0).fit_transform(X)
    labs["gmm"] = GaussianMixture(k, covariance_type="full", n_init=3, random_state=sd).fit_predict(Xg)
    labs["dec"], Zdec, info_dec = dec(X, k, seed=sd)
    print(f"metode selesai {time.time() - t0:.0f}s")
    # stabilitas antar seed per metode
    stab = {}
    for m in a.metode_stab:
        if m == "kmeans":
            ls = [kmeans(X, k, seed=sd + s) for s in range(3)]
        elif m == "gmm":
            ls = [GaussianMixture(k, covariance_type="full", n_init=3, random_state=sd + s).fit_predict(Xg) for s in range(3)]
        else:
            ls = [labs["dec"]] + [dec(X, k, seed=sd + s)[0] for s in (1, 2)]
        stab[m] = float(np.mean([adjusted_rand_score(ls[i], ls[j]) for i in range(3) for j in range(i + 1, 3)]))
    met = {}
    for m, l in labs.items():
        met[m] = {"silhouette_fusion": sil(X, l), "davies_bouldin": float(davies_bouldin_score(X, l)),
                  "nmi_mikroskop": float(normalized_mutual_info_score(l, mik)),
                  "ukuran_cluster": np.bincount(l, minlength=k).tolist(), "stabilitas_3seed": stab.get(m)}
    met["dec"]["silhouette_ruang_dec"] = sil(Zdec, labs["dec"])
    met["dec"]["iterasi"] = info_dec["iterasi"]
    met["dec"]["ARI_vs_init_kmeans"] = float(adjusted_rand_score(labs["dec"], info_dec["init_kmeans"]))
    hasil["metode"] = met
    if a.metode == "auto":
        a.metode = max(met, key=lambda m: (round(met[m]["silhouette_fusion"], 3), met[m]["stabilitas_3seed"] or 0))
        print("metode terpilih (silhouette, lalu stabilitas):", a.metode)
    hasil["ARI_antar_metode"] = {f"{p}_vs_{q}": float(adjusted_rand_score(labs[p], labs[q]))
                                 for i, p in enumerate(labs) for q in list(labs)[i + 1:]}

    # modality: image-only vs spectrum-only vs fusion (K-Means, k sama)
    li, ls_ = kmeans(Xi, k), kmeans(Xs, k)
    lf = labs[a.metode]
    hasil["modality"] = {
        "image_only": {"silhouette": sil(Xi, li), "ARI_vs_fusion": float(adjusted_rand_score(li, lf)),
                       "nmi_mikroskop": float(normalized_mutual_info_score(li, mik))},
        "spectrum_only": {"silhouette": sil(Xs, ls_), "ARI_vs_fusion": float(adjusted_rand_score(ls_, lf)),
                          "nmi_mikroskop": float(normalized_mutual_info_score(ls_, mik))},
        "ARI_image_vs_spectrum": float(adjusted_rand_score(li, ls_)),
    }
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for axi, l, nm in [(ax[0], li, "image-only"), (ax[1], ls_, "spectrum-only")]:
        ct = pd.crosstab(lf, l)
        ctn = ct.values / ct.values.sum(1, keepdims=True)
        axi.imshow(ctn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axi.set_xlabel(f"cluster {nm}"); axi.set_ylabel(f"cluster fusion ({a.metode})")
        axi.set_title(f"Isi tiap cluster fusion menurut {nm} (proporsi baris)")
        axi.set_xticks(range(ct.shape[1])); axi.set_yticks(range(ct.shape[0]))
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if ctn[i, j] > 0.05:
                    axi.text(j, i, f"{ctn[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(folder_figur(3), "modality_kontingensi.png"), dpi=120); plt.close(fig)

    # UMAP
    if not a.tanpa_umap:
        t1 = time.time()
        U = umap2d(X)
        Ui, Us = umap2d(Xi), umap2d(Xs)
        np.save(os.path.join(run, "umap_fusion.npy"), U)
        print(f"UMAP {time.time() - t1:.0f}s")
        cmap = plt.get_cmap("tab20")
        fig, ax = plt.subplots(2, 3, figsize=(18, 11))
        def sc(axi, P, c, judul, kat=True):
            for v in np.unique(c):
                m = c == v
                axi.scatter(P[m, 0], P[m, 1], s=1, alpha=.4, color=cmap(int(v) % 20) if kat else None, label=str(v))
            axi.set_title(judul); axi.set_xticks([]); axi.set_yticks([])
        sc(ax[0, 0], U, lf, f"Fusion — warna = cluster {a.metode}")
        ax[0, 0].legend(markerscale=8, fontsize=7, ncol=2)
        mk = (mik == "M2").astype(int)
        sc(ax[0, 1], U, mk, "Fusion — warna = mikroskop (0=M1, 1=M2)")
        ax[0, 1].legend(markerscale=8)
        sil_s = silhouette_samples(X, lf)
        p = ax[0, 2].scatter(U[:, 0], U[:, 1], s=1, c=sil_s, cmap="RdYlGn", vmin=-.2, vmax=.6)
        ax[0, 2].set_title("Fusion — warna = silhouette per sampel"); plt.colorbar(p, ax=ax[0, 2])
        ax[0, 2].set_xticks([]); ax[0, 2].set_yticks([])
        sc(ax[1, 0], Ui, lf, "Image-only — warna = cluster fusion")
        sc(ax[1, 1], Us, lf, "Spectrum-only — warna = cluster fusion")
        sc(ax[1, 2], Ui, mk, "Image-only — warna = mikroskop")
        fig.tight_layout(); fig.savefig(os.path.join(folder_figur(3), "umap.png"), dpi=110); plt.close(fig)
    else:
        sil_s = silhouette_samples(X, lf)

    # cluster sulit dipisah
    sil_c = pd.Series(sil_s).groupby(lf).mean()
    C = np.stack([X[lf == c].mean(0) for c in range(k)])
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    np.fill_diagonal(D, np.inf)
    pasangan = sorted({tuple(sorted((i, int(D[i].argmin())))) for i in range(k)}, key=lambda p: D[p])
    # tumpang tindih GMM: rata-rata posterior ke cluster kedua
    gm = GaussianMixture(k, covariance_type="full", random_state=0).fit(Xg, )
    post = gm.predict_proba(Xg)
    hasil["cluster_sulit"] = {
        "silhouette_per_cluster": sil_c.round(4).to_dict(),
        "pasangan_terdekat": [{"pasangan": list(map(int, p)), "jarak_pusat": float(D[p])} for p in pasangan[:5]],
        "frac_sampel_ambigu_gmm(p2>0.2)": float((np.sort(post, 1)[:, -2] > 0.2).mean()),
        "mikroskop_per_cluster_frac_M2": pd.Series(mik == "M2").groupby(lf).mean().round(3).to_dict(),
    }
    # contoh tiap cluster: 6 gambar + rata-rata spektrum
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(k, 7, figsize=(14, 1.8 * k))
    for c in range(k):
        idx = rng.choice(np.where(lf == c)[0], 6, replace=False)
        for j, i in enumerate(idx):
            ax[c, j].imshow(g8[i], cmap="gray", vmin=0, vmax=255); ax[c, j].axis("off")
        ax[c, 0].set_title(f"cluster {c} (n={int((lf == c).sum())}, sil={sil_c[c]:.2f})", fontsize=8, loc="left")
        ax[c, 6].plot(np.median(sp[lf == c], 0), lw=1); ax[c, 6].set_yticks([]); ax[c, 6].set_xticks([])
    fig.tight_layout(); fig.savefig(os.path.join(folder_figur(3), "contoh_cluster.png"), dpi=90); plt.close(fig)

    np.save(os.path.join(run, "labels.npy"), np.stack([labs["kmeans"], labs["gmm"], labs["dec"], li, ls_]))
    return hasil, lf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--tahap", choices=["banding", "k", "final"], required=True)
    ap.add_argument("--img", default="runs/t3_ssl_joint_s0/emb_img.npy")
    ap.add_argument("--spec", default="runs/t3_ssl_joint_s0/emb_spec.npy")
    ap.add_argument("--w", type=float, default=0.5, help="bobot gambar dalam fusion")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=20)
    ap.add_argument("--k_banding", type=int, nargs="+", default=[6, 10])
    ap.add_argument("--metode", choices=["kmeans", "gmm", "dec", "auto"], default="auto",
                    help="metode untuk cluster.csv (auto: silhouette fusion tertinggi, seri -> stabilitas)")
    ap.add_argument("--metode_stab", nargs="*", default=["kmeans", "gmm", "dec"])
    ap.add_argument("--tanpa_umap", action="store_true")
    ap.add_argument("--nama", default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed K-Means/GMM/DEC (stabilitas memakai seed, seed+1, seed+2)")
    a = ap.parse_args()
    set_seed(a.seed)
    run = buat_run(a.nama or f"t3_cluster_{a.tahap}", a)
    meta = pd.read_csv(os.path.join(a.data, "clustering_meta.csv"))
    mik = meta["mikroskop"].values
    z = np.load(os.path.join(a.data, "clustering.npz"))
    g8, sp = z["gambar"], z["spektrum"].astype(np.float32)

    if a.tahap == "banding":
        df = tahap_banding(a, mik, sp)
        df.round(4).to_csv(os.path.join(run, "banding_representasi.csv"), index=False)
        simpan_metrics(run, {"tabel": df.round(4).to_dict(orient="records")})
        print(df.round(3).to_string(index=False))
        return

    Xi, Xs = ruang(muat_emb(a.img)), ruang(muat_emb(a.spec))
    X = fusi(Xi, Xs, a.w)
    if a.tahap == "k":
        df, pilih = tahap_k(a, X, run, Xi, Xs)
        simpan_metrics(run, {"pilihan_k": pilih, "tabel": df.round(4).to_dict(orient="records"),
                             "img": a.img, "spec": a.spec})
        print(pilih)
        return
    hasil, lab = tahap_final(a, Xi, Xs, X, mik, g8, sp, run)
    hasil.update({"img": a.img, "spec": a.spec, "w": a.w, "metode_final": a.metode})
    simpan_metrics(run, hasil)
    pd.DataFrame({"id_sampel": meta["id_sampel"], "cluster": lab.astype(int)}).to_csv(
        os.path.join(ROOT, "cluster.csv"), index=False)
    import json
    print(json.dumps(hasil, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
