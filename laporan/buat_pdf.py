"""Gabungkan bab laporan menjadi laporan/laporan.md -> laporan/laporan.html -> laporan.pdf (root repo).

Sumber (satu-satunya tempat isi laporan ditulis; file gabungan dibuat ulang setiap kali):
    laporan/pendahuluan.md, bab_task1.md, bab_task2.md, bab_task3.md, kesimpulan.md

Langkah:
1. Bagian "## Referensi" tiap bab dikumpulkan menjadi satu "Daftar Pustaka" (tanpa duplikat).
2. Gambar utama disisipkan di akhir subbab yang membahasnya (FIGUR_UTAMA); semua gambar lain dari
   laporan_figs/ masuk "Lampiran A". Tabel lengkap angka (docs/tabel_laporan.md) disebut di lampiran.
3. Markdown -> HTML (paket `markdown`) -> PDF lewat Microsoft Edge headless.
4. Jumlah halaman dicek: badan laporan (tanpa lampiran) harus <= 18 halaman (aturan soal).

    python laporan/buat_pdf.py
"""
import glob
import os
import re
import shutil
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BAGIAN = ["pendahuluan.md", "bab_task1.md", "bab_task2.md", "bab_task3.md", "kesimpulan.md"]
BATAS_HALAMAN = 18

JUDUL = """<div class="sampul">
<h1 class="judul">Laporan UTS Deep Learning</h1>
<p class="sub">Klasifikasi Multimodal, Prediksi Remaining Useful Life, dan Clustering Multimodal</p>
<p class="anggota">Satrio Arjuna Putra · M Marcheliano Taji Suryana</p>
<p class="repo">Repositori: https://github.com/marcheliano-satrio/UTS_Deep_Learning</p>
</div>
"""

# (awal judul subbab, file gambar, keterangan) -> disisipkan di akhir subbab tersebut
FIGUR_UTAMA = [
    ("## 1c.", "task1/modality_f1_heatmap.png", "F1 per kelas untuk model gambar saja, sensor saja, dan fusi."),
    ("## 1e.", "task1/domain_fourier.png", "Analisis Fourier gambar lini A vs lini B dan garis tiruan."),
    ("## 1g.", "task1/final_kurva.png", "Kurva pembelajaran model final (3 seed)."),
    ("## 2a.", "task2/eda_regime.png", "Setelan operasi membentuk 6 regime; normalisasi per regime memperlihatkan tren degradasi."),
    ("## 2f.", "task2/uncert_kalibrasi.png", "Prediksi titik potong fold 0 dengan interval 90% dan kurva kalibrasi sebelum/sesudah conformal."),
    ("## 3e.", "task3/pilih_k.png", "Kriteria pemilihan jumlah cluster untuk k = 2–20."),
    ("## 3f.", "task3/umap.png", "UMAP ruang fusi, diwarnai cluster final."),
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #111; }
h1 { font-size: 16pt; border-bottom: 2px solid #333; padding-bottom: 3px; margin: 0 0 8px 0; page-break-before: always; }
h1.judul { page-break-before: avoid; border: none; font-size: 22pt; text-align: center; margin-top: 30px; }
.sampul { text-align: center; margin-bottom: 18px; }
.sampul .sub { font-size: 12pt; } .sampul .anggota { font-size: 11pt; font-weight: 600; } .sampul .repo { font-size: 9pt; }
.sampul + h1 { page-break-before: avoid; }
h2 { font-size: 12pt; margin: 12px 0 4px 0; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10px 0 3px 0; page-break-after: avoid; }
p, li { margin: 3px 0; text-align: justify; }
ul, ol { margin: 3px 0 3px 0; padding-left: 20px; }
table { border-collapse: collapse; margin: 6px 0; font-size: 8.4pt; page-break-inside: avoid; width: auto; }
th, td { border: 1px solid #999; padding: 2px 5px; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f3f3f3; padding: 0 2px; }
figure { margin: 8px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 62mm; }
figcaption { font-size: 8.6pt; color: #333; margin-top: 2px; }
.lampiran figure img { max-height: 95mm; }
"""


def baca(nama):
    with open(os.path.join(HERE, nama), encoding="utf-8") as f:
        return f.read()


def pisah_referensi(teks):
    """Kembalikan (teks tanpa bagian '## Referensi', daftar entri referensi)."""
    m = re.search(r"^## Referensi\s*$", teks, flags=re.M)
    if not m:
        return teks, []
    badan, ref = teks[:m.start()], teks[m.end():]
    entri, cur = [], None
    for baris in ref.splitlines():
        if baris.startswith("- "):
            cur = baris[2:].strip()
            entri.append(cur)
        elif baris.strip() and entri:
            entri[-1] += " " + baris.strip()
    return badan.rstrip() + "\n", entri


def rapikan_list(teks):
    """Python-Markdown butuh baris kosong sebelum list dan indentasi 4 spasi untuk list bersarang;
    bab ditulis dengan gaya GitHub (tanpa baris kosong, indentasi 2 spasi)."""
    out, sebelumnya = [], ""
    for baris in teks.splitlines():
        if re.match(r"^ {2}(\d+\.|-) ", baris) and out:
            baris = "  " + baris                       # list bersarang 2 -> 4 spasi
        elif re.match(r"^ {2,3}\S", baris) and re.match(r"^ {4}(\d+\.|-) |^ {4,}\S", sebelumnya):
            baris = "    " + baris.lstrip()            # lanjutan baris item bersarang
        is_list = re.match(r"^(\d+\.|-) ", baris)
        if is_list and sebelumnya.strip() and not re.match(r"^\s*(\d+\.|-) |^\s{2,}\S", sebelumnya):
            out.append("")
        out.append(baris)
        sebelumnya = baris
    return "\n".join(out) + "\n"


def figur(path_rel, ket, no):
    return (f'\n<figure><img src="../laporan_figs/{path_rel}" alt="{path_rel}">'
            f'<figcaption><b>Gambar {no}.</b> {ket} (<code>laporan_figs/{path_rel}</code>)</figcaption></figure>\n')


def sisip_figur(teks, no_awal):
    """Sisipkan FIGUR_UTAMA di akhir subbab yang judulnya diawali kunci; kembalikan (teks, terpakai, no)."""
    baris = teks.split("\n")
    no, terpakai = no_awal, []
    for kunci, path_rel, ket in FIGUR_UTAMA:
        idx = next((i for i, b in enumerate(baris) if b.startswith(kunci)), None)
        if idx is None:
            continue
        akhir = next((j for j in range(idx + 1, len(baris)) if re.match(r"^#{1,2} ", baris[j])), len(baris))
        baris.insert(akhir, figur(path_rel, ket, no))
        terpakai.append(path_rel)
        no += 1
    return "\n".join(baris), terpakai, no


def susun_md():
    potongan, referensi, terpakai, no = [], [], [], 1
    for nama in BAGIAN:
        teks, ref = pisah_referensi(baca(nama))
        teks, pakai, no = sisip_figur(rapikan_list(teks), no)
        potongan.append(teks)
        terpakai += pakai
        referensi += [r for r in ref if r not in referensi]
    utama = "\n\n".join(potongan)
    utama += "\n\n# Daftar Pustaka\n\n" + "\n".join(f"- {r}" for r in sorted(referensi)) + "\n"

    lamp = ["# Lampiran A. Gambar pendukung", "",
            "Gambar yang dirujuk di teks dengan nama filenya. Tabel lengkap seluruh angka eksperimen ada di "
            "`docs/tabel_laporan.md` (dibuat otomatis oleh `kode/tabel_laporan.py`).", ""]
    for p in sorted(glob.glob(os.path.join(ROOT, "laporan_figs", "*", "*.png"))):
        rel = os.path.relpath(p, os.path.join(ROOT, "laporan_figs")).replace("\\", "/")
        if rel in terpakai:
            continue
        lamp.append(figur(rel, f"Task {rel[4]} — {os.path.splitext(os.path.basename(rel))[0].replace('_', ' ')}.", no))
        no += 1
    return utama, "\n".join(lamp) + "\n"


def ke_html(md_teks, lampiran_md=None):
    ext = ["tables", "fenced_code", "md_in_html"]
    isi = markdown.markdown(md_teks, extensions=ext)
    if lampiran_md:
        isi += '<div class="lampiran">' + markdown.markdown(lampiran_md, extensions=ext) + "</div>"
    return (f'<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Laporan UTS Deep Learning</title>'
            f"<style>{CSS}</style></head><body>{JUDUL}{isi}</body></html>")


def cari_edge():
    for p in [shutil.which("msedge"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]:
        if p and os.path.exists(p):
            return p
    sys.exit("Microsoft Edge tidak ditemukan.")


def cetak_pdf(html_path, pdf_path):
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    subprocess.run([cari_edge(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf-no-header", f"--print-to-pdf={os.path.abspath(pdf_path)}", url],
                   check=True, capture_output=True, timeout=180)
    if not os.path.exists(pdf_path):
        sys.exit(f"PDF gagal dibuat: {pdf_path}")


def jumlah_halaman(pdf_path):
    from pypdf import PdfReader

    return len(PdfReader(pdf_path).pages)


def main():
    utama, lampiran = susun_md()
    with open(os.path.join(HERE, "laporan.md"), "w", encoding="utf-8") as f:
        f.write(utama + "\n" + lampiran)

    # badan saja (untuk cek batas halaman), lalu laporan lengkap
    tmp_html, tmp_pdf = os.path.join(HERE, "_badan.html"), os.path.join(HERE, "_badan.pdf")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(ke_html(utama))
    cetak_pdf(tmp_html, tmp_pdf)
    n_badan = jumlah_halaman(tmp_pdf)
    os.remove(tmp_html); os.remove(tmp_pdf)

    html_path, pdf_path = os.path.join(HERE, "laporan.html"), os.path.join(ROOT, "laporan.pdf")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(ke_html(utama, lampiran))
    cetak_pdf(html_path, pdf_path)
    n_total = jumlah_halaman(pdf_path)
    status = "OK" if n_badan <= BATAS_HALAMAN else f"MELEBIHI BATAS {BATAS_HALAMAN}"
    print(f"ditulis: {pdf_path}\nhalaman badan (tanpa lampiran): {n_badan} [{status}] | total: {n_total}")


if __name__ == "__main__":
    main()
