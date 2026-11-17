"""Task 2d — tiga arsitektur sekuens untuk regresi RUL.

Input: (B, W, F+1), kanal terakhir = mask (1 = siklus asli, 0 = padding kiri).
Output: RUL ternormalisasi (RUL/150). Semua model punya dropout sehingga bisa dipakai untuk
MC dropout saat inferensi.

- gru         : GRU 2 lapis (hidden 64). Rekuren, kausal secara alami; status terakhir dipakai.
- tcn         : Temporal Convolutional Network: konvolusi kausal ber-dilasi (1,2,4,8) dengan
                residual. Receptive field 1 + 2*(k-1)*sum(dilasi) = 61 >= W.
- transformer : encoder Transformer 2 lapis (d=64, 4 head), positional embedding dipelajari,
                padding di-mask; token terakhir (siklus saat ini) dipakai sebagai ringkasan.
                Isi window hanya siklus <= t, jadi tidak ada kebocoran masa depan.
"""
import torch
import torch.nn as nn


class GRUNet(nn.Module):
    def __init__(self, f, h=64, p=0.2):
        super().__init__()
        self.gru = nn.GRU(f, h, num_layers=2, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(h, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(1)


class CausalBlock(nn.Module):
    def __init__(self, c, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c, c, k, dilation=d)
        self.c2 = nn.Conv1d(c, c, k, dilation=d)
        self.drop = nn.Dropout(p)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.drop(self.act(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(self.act(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return self.act(x + y)


class TCNNet(nn.Module):
    def __init__(self, f, c=64, k=3, dil=(1, 2, 4, 8), p=0.2):
        super().__init__()
        self.inp = nn.Conv1d(f, c, 1)
        self.blocks = nn.Sequential(*[CausalBlock(c, k, d, p) for d in dil])
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(c, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, x):
        y = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(y[:, :, -1]).squeeze(1)


class TransformerNet(nn.Module):
    def __init__(self, f, d=64, n_head=4, n_layer=2, W=50, p=0.2):
        super().__init__()
        self.inp = nn.Linear(f, d)
        self.pos = nn.Parameter(torch.zeros(1, W, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_head, 2 * d, p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, x):
        W = x.shape[1]
        pad = x[:, :, -1] < 0.5                  # True = padding
        # Tidak perlu causal mask: seluruh isi window sudah <= siklus saat ini (tidak ada
        # masa depan). Token terakhir selalu data asli, jadi tiap baris punya key yang valid.
        h = self.enc(self.inp(x) + self.pos[:, -W:], src_key_padding_mask=pad)
        return self.head(h[:, -1]).squeeze(1)


def buat_model(nama, f, W, p=0.2):
    if nama == "gru":
        return GRUNet(f, p=p)
    if nama == "tcn":
        return TCNNet(f, p=p)
    if nama == "transformer":
        return TransformerNet(f, W=max(W, 50), p=p)
    raise ValueError(nama)


def aktifkan_mc_dropout(model):
    """Eval mode untuk semua layer, kecuali Dropout tetap aktif (MC dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


if __name__ == "__main__":
    for n in ["gru", "tcn", "transformer"]:
        m = buat_model(n, 22, 30)
        x = torch.randn(4, 30, 22)
        x[:, :, -1] = 1
        x[0, :10, -1] = 0
        y = m(x)
        assert y.shape == (4,) and torch.isfinite(y).all()
        print(n, sum(p.numel() for p in m.parameters()))
