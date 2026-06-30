# Ringkasan Pembuatan Dataset — Spell Correction Bahasa Indonesia

Dokumen ini merangkum **cara pembuatan** dan **sumber data** dari empat dataset yang
dihasilkan oleh `dataset.ipynb`, untuk sistem koreksi ejaan (spell correction) Bahasa Indonesia.

---

## 1. Gambaran Umum

Notebook membangkitkan 4 dataset sekaligus dengan jumlah mengikuti target ideal:

| Dataset | File | Fungsi | Jumlah Ideal | Hasil Akhir |
|---|---|---|---|---|
| **Kamus** | `dataset/dictionary/kamus.txt` | Dictionary Checking & Edit Distance | 50.000–100.000 kata | **100.000 kata** |
| **Corpus** | `dataset/corpus/corpus.txt` | Bigram & Trigram Language Model | 50.000–100.000 kalimat | **96.715 kalimat** |
| **Typo Dataset** | `dataset/typo_dataset/typo_dataset.csv` | Fine-tuning IndoBERT | 20.000–50.000 pasangan | **30.000 pasangan** |
| **Test Dataset** | `dataset/test_dataset/test.csv` | Evaluasi (Acc, P, R, F1, WER) | 2.000–5.000 pasangan | **3.000 pasangan** |

Prinsip penting:
- **Reproducible** — memakai `SEED = 42` agar hasil acak konsisten.
- **Fallback otomatis** — bila unduhan gagal/offline, notebook beralih ke generator internal supaya target tetap terpenuhi.
- **Tanpa data leakage** — kalimat untuk training dan testing diambil dari kolam (pool) yang **terpisah (disjoint)**.

---

## 2. Setup Awal (Cell Setup)

1. Import library: `os`, `re`, `csv`, `io`, `tarfile`, `random`, `pathlib`, dan `requests` (opsional).
2. Jika `requests` tidak tersedia → masuk **mode offline** (pakai fallback generator).
3. Membuat struktur folder otomatis di bawah `dataset/`:
   ```
   dataset/
   ├── dictionary/      → kamus.txt
   ├── corpus/          → corpus.txt
   ├── typo_dataset/    → typo_dataset.csv
   └── test_dataset/    → test.csv
   ```
4. Dua fungsi unduh utama:
   - `fetch_text(url)` — mengunduh berkas teks (untuk kamus).
   - `fetch_bytes(url)` — mengunduh berkas biner `.tar.gz` (untuk corpus).
   - Keduanya memakai header `User-Agent: Mozilla/5.0` agar tidak ditolak server.

---

## 3. Dataset Kamus (`kamus.txt`)

**Tujuan:** daftar kata unik (lowercase, satu kata per baris) untuk *Dictionary Checking*,
*Candidate Generation*, dan *Edit Distance*. Target tepat **100.000 kata**.

### Sumber (online, terverifikasi 200 OK)
| Sumber | URL | Jumlah |
|---|---|---|
| `geovedi/indonesian-wordlist` (gabungan KBBI 2001, crawl web, kamus, myspell) | `raw.githubusercontent.com/geovedi/indonesian-wordlist/master/00-indonesian-wordlist.lst` | ~78.700 kata |
| Hunspell `id_ID` dari `LibreOffice/dictionaries` | `raw.githubusercontent.com/LibreOffice/dictionaries/master/id/id_ID.dic` | ~43.254 kata |

Total kata asli unik setelah digabung: **~89.300 kata**.

### Cara pembuatan
1. **Unduh** kedua sumber lalu **bersihkan** dengan `clean_words()`:
   - Ubah ke lowercase, ambil token pertama.
   - Buang flag hunspell (`makan/A` → `makan`).
   - Hanya simpan kata berisi huruf saja (`^[a-z]+$`) dan panjang ≥ 2.
   - Disimpan dalam `set` agar otomatis unik.
2. **Top-up afiksasi** — karena sumber online hanya ~89.300 kata, sisanya (~10.700) dilengkapi
   **bentuk berimbuhan yang sah** dari kata asli (panjang 4–9 huruf), dengan kombinasi:
   - **Prefix:** `me, ber, di, ter, pe, se, ke, meng, mem, men, peng`
   - **Suffix:** `an, kan, i, nya, lah, kah`
   sampai tepat 100.000 kata.
3. **Fallback offline:** tersedia `KATA_DASAR` (kumpulan kata dasar bawaan) bila benar-benar tidak ada koneksi.
4. Tulis hasil terurut ke `kamus.txt` (~1 MB).

---

## 4. Dataset Corpus (`corpus.txt`)

**Tujuan:** kumpulan kalimat (satu kalimat per baris) untuk membangun **Bigram & Trigram Language Model**.
Target ideal **50.000–100.000 kalimat**.

### Sumber (online)
**Leipzig Corpora Collection** — kalimat berita Bahasa Indonesia asli. Ukuran default `100K` (arsip ~23 MB).
Beberapa varian dicoba berurutan:
```
https://downloads.wortschatz-leipzig.de/corpora/ind_news_2022_100K.tar.gz   ← berhasil dipakai
https://downloads.wortschatz-leipzig.de/corpora/ind_mixed_2013_100K.tar.gz
https://downloads.wortschatz-leipzig.de/corpora/ind_wikipedia_2021_100K.tar.gz
https://downloads.wortschatz-leipzig.de/corpora/ind_news_2008_100K.tar.gz
```

### Cara pembuatan
1. Unduh arsip `.tar.gz`, lalu `extract_leipzig_sentences()` mengambil file `*-sentences.txt`
   (format `id<TAB>kalimat`) dan mengambil kolom kalimatnya.
2. Filter kualitas dengan `is_good_sentence()`:
   - Panjang **3–30 kata**.
   - Mengandung huruf.
   - Rasio huruf > 60% dari total karakter (membuang kalimat penuh angka/simbol).
3. Hapus duplikat (`dict.fromkeys`) dan batasi maksimal 100.000 kalimat.
   Hasil dari `ind_news_2022_100K`: **96.715 kalimat layak**.
4. **Fallback generator** — bila online < target, tambah kalimat sintetis berbasis template
   (Subjek + Predikat + Objek + Keterangan).
5. Acak urutan lalu tulis ke `corpus.txt` (~10,7 MB).

---

## 5. Typo Generator (dasar pembuatan pasangan)

Membangkitkan kesalahan ketik dari kalimat benar di corpus. **Empat operasi level karakter:**

| Operasi | Contoh | Fungsi |
|---|---|---|
| **Insertion** | `makan` → `makann` | `typo_insertion()` — sisip huruf / gandakan huruf |
| **Deletion** | `pergi` → `peri` | `typo_deletion()` — hapus huruf |
| **Substitution** | `buku` → `biku` | `typo_substitution()` — ganti huruf (berdasarkan tetangga keyboard QWERTY agar realistis) |
| **Transposition** | `buku` → `ubku` | `typo_transposition()` — tukar dua huruf bersebelahan |

- `make_typo_sentence()` memilih beberapa kata (≥ 3 huruf), merusaknya, sambil menjaga
  tanda baca/awalan-akhiran non-huruf tetap utuh.

---

## 6. Dataset Typo & Test (`typo_dataset.csv` & `test.csv`)

**Tujuan:** pasangan kalimat `error,correct` — typo dataset untuk fine-tuning IndoBERT,
test dataset untuk evaluasi.

### Pembagian tanpa data leakage
Corpus diacak lalu dibagi menjadi dua kolam **disjoint**:
- ~15% kalimat (atau 2× jumlah test) → **kolam test** (14.507 kalimat).
- Sisanya → **kolam train** (82.208 kalimat).

Dengan begitu kalimat uji tidak pernah dilihat saat pelatihan.

### Cara pembuatan
1. `generate_pairs()` mengambil kalimat dari masing-masing kolam, merusaknya dengan
   `make_typo_sentence()`. Jumlah error per kalimat dipilih acak: **1 (60%), 2 (30%), 3 (10%)**.
2. Hasil:
   - `typo_dataset.csv` → **30.000 pasangan** (dari kolam train).
   - `test.csv` → **3.000 pasangan** (dari kolam test).
3. `write_pairs()` menulis CSV dengan header kolom `error,correct`.

---

## 7. Verifikasi Akhir

Cell terakhir menghitung jumlah baris tiap file dan membandingkan dengan rentang ideal:

```
Kamus        : 100,000 kata     ✓ ideal (50,000-100,000)
Corpus       :  96,715 kalimat  ✓ ideal (50,000-100,000)
Typo dataset :  30,000 pasangan ✓ ideal (20,000-50,000)
Test dataset :   3,000 pasangan ✓ ideal (2,000-5,000)
```

Semua dataset memenuhi target ideal.

---

## 8. Ringkasan Sumber Data

| Dataset | Sumber Utama | Metode |
|---|---|---|
| Kamus | `geovedi/indonesian-wordlist` (GitHub) + Hunspell `id_ID` LibreOffice (GitHub) | Unduh + bersihkan + top-up afiksasi |
| Corpus | Leipzig Corpora Collection — `ind_news_2022_100K` | Unduh `.tar.gz` + filter kualitas |
| Typo Dataset | Diturunkan dari corpus (kolam train) | Typo generator 4 operasi |
| Test Dataset | Diturunkan dari corpus (kolam test, disjoint) | Typo generator 4 operasi |

> **Catatan:** Semua sumber bersifat publik dan terverifikasi. Jika offline, notebook tetap
> menghasilkan dataset lengkap melalui generator internal (kata dasar + template kalimat).
