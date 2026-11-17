"""Task 3b-c — representasi tanpa label: SimCLR gambar, SimCLR spektrum, gabungan, dan
embedding pretrained ResNet18 (ImageNet) sebagai pembanding.

SimCLR (Chen dkk., 2020): dua augmentasi acak dari sampel yang sama harus punya embedding
mirip (pasangan positif), sampel lain di batch menjadi negatif (loss NT-Xent). Yang dipelajari
adalah fitur yang TIDAK berubah oleh augmentasi. Karena itu augmentasi dipilih untuk meniru
distorsi pengganggu yang ditemukan di EDA (task3_eda.py) — bukan perbedaan material:

Gambar
- rotasi sudut bebas 0-360 + flip : orientasi tekstur tersebar merata 0-180 derajat
                                    (rotasi = distorsi akuisisi, bukan ciri material)
- skala 0.75-1.33 + geser kecil   : distorsi skala / perbesaran mikroskop
- kecerahan & kontras acak        : M2 rata-rata lebih terang (146 vs 128) dari M1
- blur Gaussian / noise acak      : M1 lebih blur (energi Laplacian 14.7 vs 20.7 di M2)
Spektrum
- geser +-6 titik                 : posisi puncak bergeser antar sampel (distorsi shift)
- baseline acak (offset+miring+lengkung) : kemiringan baseline bervariasi lebar
- spike sintetis 0-2 titik        : ~22% spektrum punya spike tunggal
- noise & skala amplitudo kecil   : variasi intensitas pengukuran

Mode (--modal):
  img        : SimCLR gambar saja
  spec       : SimCLR spektrum saja
  joint      : kedua encoder dilatih bersama: NT-Xent gambar + NT-Xent spektrum + NT-Xent
               lintas modality (gambar_i <-> spektrum_i, seperti CLIP), sehingga ruang embedding
               kedua modality selaras (fusion tingkat representasi)
  pretrained : ResNet18 bobot ImageNet dari torchvision (He dkk., 2016;
               https://download.pytorch.org/models/resnet18-f37072fd.pth), tanpa fine-tuning.
Embedding akhir = keluaran encoder (sebelum projection head). Untuk gambar dirata-rata atas
8 transformasi dihedral (rotasi 90 derajat x flip) supaya lebih invarian rotasi.

  python kode/task3_cluster/task3_ssl.py --modal img --nama t3_ssl_img
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# folder task ini (modul task) + kode/ (paket common) masuk sys.path
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from common import buat_run, get_device, set_seed, simpan_json, simpan_metrics  # noqa: E402


# ------------------------------------------------------------------ encoder
def blok2d(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


def blok1d(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
                         nn.Conv1d(cout, cout, 5, padding=2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool1d(2))


class EncGambar(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(blok2d(1, 32), blok2d(32, 64), blok2d(64, 128), blok2d(128, 256))

    def forward(self, x):
        return self.net(x).mean((2, 3))


class EncSpektrum(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(blok1d(1, 32), blok1d(32, 64), blok1d(64, 128), blok1d(128, 256))

    def forward(self, x):
        return self.net(x[:, None]).mean(2)


def proyeksi(d=256, out=128):
    return nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True), nn.Linear(d, out))


@torch.autocast("cuda", enabled=False)
def nt_xent(z1, z2, tau):
    """Loss NT-Xent: pasangan (z1_i, z2_i) positif, 2N-2 sampel lain negatif. Dihitung di fp32."""
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.autocast("cuda", enabled=False)
def clip_loss(a, b, tau):
    """Kontras lintas modality: gambar_i harus paling mirip spektrum_i (dua arah). fp32."""
    a, b = F.normalize(a.float(), dim=1), F.normalize(b.float(), dim=1)
    logit = a @ b.t() / tau
    t = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logit, t) + F.cross_entropy(logit.t(), t))


# ------------------------------------------------------------------ augmentasi (GPU, per sampel)
def aug_gambar(x):
    """x: (B,1,64,64) ternormalisasi. Rotasi bebas, skala, geser, flip, kecerahan/kontras, blur, noise."""
    b, dev = x.shape[0], x.device
    th = torch.rand(b, device=dev) * 2 * math.pi
    sk = torch.exp(torch.empty(b, device=dev).uniform_(math.log(0.75), math.log(1.33)))
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(th) / sk, torch.sin(th) / sk
    A = torch.zeros(b, 2, 3, device=dev)
    A[:, 0, 0], A[:, 0, 1] = cos * flip, -sin
    A[:, 1, 0], A[:, 1, 1] = sin * flip, cos
    A[:, :, 2] = torch.empty(b, 2, device=dev).uniform_(-0.15, 0.15)
    grid = F.affine_grid(A, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    # blur Gaussian per sampel (sigma 0.3-1.5) untuk 50% sampel
    sig = torch.empty(b, device=dev).uniform_(0.3, 1.5) * (torch.rand(b, device=dev) < 0.5)
    r = torch.arange(-3, 4, device=dev).float()
    k1 = torch.exp(-r[None] ** 2 / (2 * sig[:, None].clamp(min=1e-3) ** 2))
    k1 = torch.where(sig[:, None] > 0, k1, (r[None] == 0).float())
    k1 = k1 / k1.sum(1, keepdim=True)
    k2 = (k1[:, :, None] * k1[:, None, :])[:, None]                      # (B,1,7,7)
    x = F.conv2d(F.pad(x.transpose(0, 1), (3, 3, 3, 3), mode="reflect"), k2, groups=b).transpose(0, 1)
    c = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    br = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.6, 0.6)
    x = x * c + br
    x = x + torch.randn_like(x) * torch.empty(b, 1, 1, 1, device=dev).uniform_(0, 0.25)
    return x


def aug_spektrum(s):
    """s: (B,128) ternormalisasi. Geser, baseline acak, spike sintetis, noise, skala amplitudo."""
    b, L = s.shape
    dev = s.device
    geser = torch.randint(-6, 7, (b,), device=dev)
    idx = (torch.arange(L, device=dev)[None] - geser[:, None]).clamp(0, L - 1)
    s = torch.gather(s, 1, idx)
    t = torch.linspace(-1, 1, L, device=dev)[None]
    a0 = torch.empty(b, 1, device=dev).uniform_(-0.5, 0.5)
    a1 = torch.empty(b, 1, device=dev).uniform_(-0.8, 0.8)
    a2 = torch.empty(b, 1, device=dev).uniform_(-0.4, 0.4)
    s = s * torch.empty(b, 1, device=dev).uniform_(0.8, 1.2) + a0 + a1 * t + a2 * t ** 2
    n_sp = torch.randint(0, 3, (b,), device=dev)
    for j in range(2):
        pos = torch.randint(0, L, (b,), device=dev)
        amp = torch.empty(b, device=dev).uniform_(2, 8) * (n_sp > j)
        s = s.scatter_add(1, pos[:, None], amp[:, None])
    return s + torch.randn_like(s) * 0.05


# ------------------------------------------------------------------ embedding
@torch.no_grad()
def embed_gambar(enc, g, bs=1024, tta=True):
    enc.eval()
    out = []
    for i in range(0, len(g), bs):
        x = g[i:i + bs]
        vs = []
        for f in ([False, True] if tta else [False]):
            xf = x.flip(3) if f else x
            for k in (range(4) if tta else [0]):
                vs.append(enc(torch.rot90(xf, k, (2, 3))))
        out.append(torch.stack(vs).mean(0).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def embed_spektrum(enc, s, bs=4096):
    enc.eval()
    return torch.cat([enc(s[i:i + bs]).float().cpu() for i in range(0, len(s), bs)]).numpy()


@torch.no_grad()
def embed_pretrained(g_uint8, dev, bs=256):
    """ResNet18 ImageNet (torchvision), avgpool 512-d, TTA 8 dihedral."""
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
    m.fc = nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
    out = []
    for i in range(0, len(g_uint8), bs):
        x = torch.from_numpy(g_uint8[i:i + bs]).to(dev).float()[:, None] / 255.0
        x = F.interpolate(x, size=112, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
        x = (x - mean) / std
        vs = []
        for f in [False, True]:
            xf = x.flip(3) if f else x
            for k in range(4):
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    vs.append(m(torch.rot90(xf, k, (2, 3))).float())
        out.append(torch.stack(vs).mean(0).cpu())
    return torch.cat(out).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--modal", choices=["img", "spec", "joint", "pretrained"], default="img")
    ap.add_argument("--nama", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--lam_cross", type=float, default=0.5, help="bobot loss lintas modality (joint)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    a.nama = a.nama or f"t3_ssl_{a.modal}_s{a.seed}" + ("_smoke" if a.smoke else "")
    set_seed(a.seed)
    dev = get_device()
    run = buat_run(a.nama, a)
    t0 = time.time()
    z = np.load(os.path.join(a.data, "clustering.npz"))
    g8, sp = z["gambar"], z["spektrum"].astype(np.float32)
    if a.smoke:
        g8, sp, a.epochs = g8[:1024], sp[:1024], 2

    if a.modal == "pretrained":
        e = embed_pretrained(g8, dev)
        np.save(os.path.join(run, "emb_img.npy"), e.astype(np.float32))
        simpan_metrics(run, {"dim": e.shape[1], "waktu_detik": round(time.time() - t0, 1),
                             "sumber_bobot": "torchvision ResNet18_Weights.IMAGENET1K_V1"})
        print(f"[{a.nama}] embedding {e.shape} {time.time() - t0:.0f}s")
        return

    # normalisasi global (statistik seluruh data; tidak ada label sehingga tidak ada kebocoran)
    g_mu, g_sd = float(g8.mean() / 255), float(g8.std() / 255)
    s_mu, s_sd = float(sp.mean()), float(sp.std())
    G = ((torch.from_numpy(g8).to(dev).float()[:, None] / 255) - g_mu) / g_sd
    S = (torch.from_numpy(sp).to(dev) - s_mu) / s_sd
    pakai_g, pakai_s = a.modal in ("img", "joint"), a.modal in ("spec", "joint")
    mods = nn.ModuleDict()
    if pakai_g:
        mods["enc_g"], mods["proj_g"] = EncGambar(), proyeksi()
    if pakai_s:
        mods["enc_s"], mods["proj_s"] = EncSpektrum(), proyeksi()
    if a.modal == "joint":
        mods["cross_g"], mods["cross_s"] = proyeksi(), proyeksi()
    mods = mods.to(dev)
    opt = torch.optim.AdamW(mods.parameters(), lr=a.lr, weight_decay=1e-4)
    n = len(g8)
    a.bs = min(a.bs, n)            # subset kecil (--smoke / data_kecil) bisa lebih kecil dari 1 batch
    per_ep = max(1, n // a.bs)     # batch sisa dibuang: NT-Xent butuh batch penuh (banyak negatif)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * per_ep, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    gen = torch.Generator().manual_seed(a.seed)
    hist = []
    for ep in range(1, a.epochs + 1):
        mods.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        tot = {"g": 0.0, "s": 0.0, "x": 0.0}
        for i in range(per_ep):
            b = perm[i * a.bs:(i + 1) * a.bs]
            loss = 0
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                if pakai_g:
                    hg1, hg2 = mods["enc_g"](aug_gambar(G[b])), mods["enc_g"](aug_gambar(G[b]))
                    lg = nt_xent(mods["proj_g"](hg1).float(), mods["proj_g"](hg2).float(), a.tau)
                    loss = loss + lg; tot["g"] += lg.item()
                if pakai_s:
                    hs1, hs2 = mods["enc_s"](aug_spektrum(S[b])), mods["enc_s"](aug_spektrum(S[b]))
                    ls = nt_xent(mods["proj_s"](hs1).float(), mods["proj_s"](hs2).float(), a.tau)
                    loss = loss + ls; tot["s"] += ls.item()
                if a.modal == "joint":
                    lx = clip_loss(mods["cross_g"](hg1).float(), mods["cross_s"](hs1).float(), a.tau)
                    loss = loss + a.lam_cross * lx; tot["x"] += lx.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        hist.append({"epoch": ep, **{f"loss_{k}": v / per_ep for k, v in tot.items() if v}})
        if ep % 10 == 0 or ep == 1 or ep == a.epochs:
            print(f"[{a.nama}] ep {ep:3d} {hist[-1]} {time.time() - t0:.0f}s")

    m = {"epochs": a.epochs, "loss_akhir": hist[-1], "waktu_detik": round(time.time() - t0, 1)}
    if pakai_g:
        e = embed_gambar(mods["enc_g"], G)
        np.save(os.path.join(run, "emb_img.npy"), e.astype(np.float32))
        torch.save(mods["enc_g"].state_dict(), os.path.join(run, "enc_img.pt"))
        m["dim_img"] = e.shape[1]
    if pakai_s:
        e = embed_spektrum(mods["enc_s"], S)
        np.save(os.path.join(run, "emb_spec.npy"), e.astype(np.float32))
        torch.save(mods["enc_s"].state_dict(), os.path.join(run, "enc_spec.pt"))
        m["dim_spec"] = e.shape[1]
    simpan_json(hist, os.path.join(run, "history.json"))
    simpan_json({"g_mu": g_mu, "g_sd": g_sd, "s_mu": s_mu, "s_sd": s_sd}, os.path.join(run, "norm.json"))
    simpan_metrics(run, m)
    print(f"[{a.nama}] selesai {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
