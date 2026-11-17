"""Task 1 — encoder gambar, encoder sinyal, dan tiga strategi fusion.

Encoder gambar: CNN kecil 4 blok (conv-BN-ReLU x2 + maxpool), 64x64 -> peta fitur 4x4x256.
  Alasan: gambar kecil & grayscale; cacat (gores, retak, noda, titik kontaminasi) adalah pola
  lokal sehingga CNN dengan bias induktif lokal cukup, dilatih dari nol (tanpa data eksternal).
Encoder sinyal: 1D-CNN 4 blok, input 6 kanal (3 nilai + 3 mask), 128 -> 8 token x 256.
  Alasan: cacat sensor (mis. las_dingin = plateau arus lebih rendah) berbentuk pola lokal
  dalam waktu; 1D-CNN cepat dan stabil. Kanal mask memberi tahu model kanal mana yang hilang.

Fusion (--fusion):
- concat : intermediate/feature-level fusion. [v_img, v_sen * ada, mask] -> MLP.
- late   : tiap modality punya kepala klasifikasi sendiri; logit = l_img + a_k * ada * l_sen
           (a_k bobot per kelas yang dipelajari). Kalau sensor hilang, otomatis hanya gambar.
- xattn  : cross-attention dua arah antara 16 token gambar dan 8 token sinyal (+1 token
           "kosong" yang dipelajari, supaya attention tetap terdefinisi saat sensor hilang).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def blok2d(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


def blok1d(cin, cout):
    return nn.Sequential(
        nn.Conv1d(cin, cout, 5, padding=2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
        nn.Conv1d(cout, cout, 5, padding=2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
        nn.MaxPool1d(2))


class ImageEncoder(nn.Module):
    def __init__(self, c=(32, 64, 128, 256)):
        super().__init__()
        self.net = nn.Sequential(blok2d(1, c[0]), blok2d(c[0], c[1]), blok2d(c[1], c[2]), blok2d(c[2], c[3]))
        self.dim = c[-1]

    def forward(self, x):                     # (B,1,64,64)
        f = self.net(x)                       # (B,256,4,4)
        return f.mean((2, 3)), f.flatten(2).transpose(1, 2)   # vektor (B,256), token (B,16,256)


class SignalEncoder(nn.Module):
    def __init__(self, c=(32, 64, 128, 256)):
        super().__init__()
        self.net = nn.Sequential(blok1d(6, c[0]), blok1d(c[0], c[1]), blok1d(c[1], c[2]), blok1d(c[2], c[3]))
        self.dim = c[-1]

    def forward(self, sen, mask):             # sen (B,3,128), mask (B,3)
        x = torch.cat([sen * mask[:, :, None], mask[:, :, None].expand_as(sen)], 1)
        f = self.net(x)                       # (B,256,8)
        return f.mean(2), f.transpose(1, 2)   # (B,256), token (B,8,256)


def kepala(din, n_kelas, p=0.3):
    return nn.Sequential(nn.Linear(din, 256), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(256, n_kelas))


class CrossAttnFusion(nn.Module):
    def __init__(self, d=256, n_head=4, n_layer=1, p=0.1):
        super().__init__()
        self.pos_img = nn.Parameter(torch.zeros(1, 16, d))
        self.pos_sen = nn.Parameter(torch.zeros(1, 8, d))
        self.kosong = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos_img, std=0.02)
        nn.init.trunc_normal_(self.pos_sen, std=0.02)
        nn.init.trunc_normal_(self.kosong, std=0.02)
        self.i2s = nn.ModuleList([nn.MultiheadAttention(d, n_head, dropout=p, batch_first=True) for _ in range(n_layer)])
        self.s2i = nn.ModuleList([nn.MultiheadAttention(d, n_head, dropout=p, batch_first=True) for _ in range(n_layer)])
        self.ln_i = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])
        self.ln_s = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])

    def forward(self, ti, ts, ada):           # ada: (B,) 1 jika ada kanal sensor terekam
        b = ti.shape[0]
        ti = ti + self.pos_img
        ts = torch.cat([self.kosong.expand(b, -1, -1), ts + self.pos_sen], 1)       # (B,9,d)
        pad = torch.zeros(b, ts.shape[1], dtype=torch.bool, device=ti.device)
        pad[:, 1:] = ~ada.bool()[:, None]     # token sensor diabaikan jika sensor hilang
        for a1, a2, l1, l2 in zip(self.i2s, self.s2i, self.ln_i, self.ln_s):
            ti = l1(ti + a1(ti, ts, ts, key_padding_mask=pad, need_weights=False)[0])
            ts = l2(ts + a2(ts, ti, ti, need_weights=False)[0])
        vs = (ts[:, 1:] * (~pad[:, 1:, None])).sum(1) / (~pad[:, 1:]).sum(1, keepdim=True).clamp(min=1)
        return torch.cat([ti.mean(1), vs], 1)


class Task1Net(nn.Module):
    def __init__(self, modal="fusion", fusion="concat", n_kelas=7):
        super().__init__()
        self.modal, self.fusion = modal, fusion
        if modal in ("image", "fusion"):
            self.enc_i = ImageEncoder()
        if modal in ("sensor", "fusion"):
            self.enc_s = SignalEncoder()
        if modal == "image":
            self.head = kepala(256, n_kelas)
        elif modal == "sensor":
            self.head = kepala(256 + 3, n_kelas)
        elif fusion == "concat":
            self.head = kepala(256 + 256 + 3, n_kelas)
        elif fusion == "late":
            self.head_i = kepala(256, n_kelas)
            self.head_s = kepala(256 + 3, n_kelas)
            self.a = nn.Parameter(torch.ones(n_kelas))
        elif fusion == "xattn":
            self.xattn = CrossAttnFusion()
            self.head = kepala(512 + 3, n_kelas)
        else:
            raise ValueError(fusion)

    def forward(self, img, sen, mask):
        ada = (mask.sum(1) > 0).float()
        if self.modal == "image":
            return self.head(self.enc_i(img)[0])
        vs, ts = self.enc_s(sen, mask)
        if self.modal == "sensor":
            return self.head(torch.cat([vs * ada[:, None], mask], 1))
        vi, ti = self.enc_i(img)
        if self.fusion == "concat":
            return self.head(torch.cat([vi, vs * ada[:, None], mask], 1))
        if self.fusion == "late":
            ls = self.head_s(torch.cat([vs, mask], 1))
            return self.head_i(vi) + self.a * ada[:, None] * ls
        return self.head(torch.cat([self.xattn(ti, ts, ada), mask], 1))


def jumlah_parameter(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for modal, fusion in [("image", "-"), ("sensor", "-"), ("fusion", "concat"), ("fusion", "late"), ("fusion", "xattn")]:
        m = Task1Net(modal, fusion if modal == "fusion" else "concat")
        x = torch.randn(4, 1, 64, 64)
        s = torch.randn(4, 3, 128)
        mk = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 1]]).float()
        out = m(x, s, mk)
        assert out.shape == (4, 7) and torch.isfinite(out).all()
        print(f"{modal:7s} {fusion:6s} params={jumlah_parameter(m):,}")
