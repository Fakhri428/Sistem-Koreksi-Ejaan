# Ringkasan Pembuatan Model — Spell Correction Bahasa Indonesia

Dokumen ini merangkum **cara pembuatan sistem koreksi ejaan** pada notebook
`02_spell_correction_hybrid_indobert.ipynb`. Sistem menggabungkan tiga komponen:
**Edit Distance**, **N-Gram Language Model**, dan re-ranking **IndoBERT**.
(Pembuatan datasetnya dibahas terpisah di `ringkasan_pembuatan_dataset.md`.)

---

## 1. Gambaran Umum

Notebook membangun pipeline koreksi ejaan dengan pendekatan **noisy-channel
sederhana**: untuk setiap kata yang dianggap salah, sistem membangkitkan kandidat
koreksi lalu memilih kandidat terbaik berdasarkan skor gabungan
**model bahasa (konteks) − penalti jarak edit**.

Tiga varian model dibangun untuk *baseline comparison*:

| Model | Metode | Pemilihan kandidat |
|---|---|---|
| **A — Edit Distance** | Jarak edit terkecil | Seri dipecah frekuensi unigram |
| **B — N-Gram** | Edit Distance + bigram LM | Skor konteks − penalti edit |
| **C — Hybrid** | N-Gram (Top-K) + re-ranking IndoBERT | Skor N-Gram + IndoBERT − penalti edit |

Empat masukan dataset dipakai: `kamus.txt` (dictionary checking & candidate
generation), `corpus.txt` (membangun N-Gram), `typo_dataset.csv` (fine-tuning
IndoBERT), dan `test.csv` (evaluasi).

---

## 2. Import & Memuat Dataset (Bagian 1–2)

1. Import library standar (`re`, `math`, `csv`, `pickle`, `collections.Counter`)
   serta `torch` + `transformers` untuk IndoBERT.
2. Memuat **kamus** menjadi `set` (`dictionary`) untuk pengecekan O(1).
3. Memuat **corpus** sebagai daftar kalimat untuk membangun N-Gram.
4. Menetapkan lokasi keluaran model di `models/`.

---

## 3. Preprocessing & Dictionary Checking (Bagian 3–4)

- **Tokenizer berbasis regex** (`tokenize`) — bekerja offline tanpa unduhan
  tambahan; memisahkan kata dari tanda baca.
- `split_tokens()` / `match_case()` — menjaga tanda baca, spasi, dan **kapitalisasi
  asli** saat menyusun kembali kalimat hasil koreksi.
- `is_valid(word)` — kata dianggap benar bila ada di kamus (dictionary checking).
  Kata yang valid **tidak dikoreksi**.

---

## 4. Edit Distance & Candidate Generation (Bagian 5–6)

- **`levenshtein(a, b)`** — jarak edit (insert/delete/substitute/transpose level
  karakter) sebagai ukuran kemiripan ejaan.
- **Candidate Generation gaya Norvig:**
  - `edits1(word)` — semua kata berjarak edit 1 (deletes, transposes, replaces, inserts).
  - `edits2(word)` — semua kata berjarak edit 2.
  - `candidates(word)` — **union** `known(edits1) ∪ known(edits2)`, yaitu hanya
    kandidat yang **ada di kamus**.

> **Catatan eksperimen (lihat `eval_classic.py` / `sweep.py`):** seluruh typo pada
> dataset berjarak edit 1–2, sehingga perluasan ke `edits3`, keyboard-adjacency,
> atau pool lebih besar **tidak** menaikkan recall — plafon recall ditentukan oleh
> cakupan kamus (~8,8% kata benar berada di luar kamus / OOV). Union (bukan
> short-circuit) dipakai agar koreksi jarak-2 tidak hilang saat sudah ada
> kandidat jarak-1.

---

## 5. N-Gram Language Model (Bagian 7–8)

1. Membangun **unigram** dan **bigram** `Counter` dari corpus, dengan penanda
   awal/akhir kalimat `<s>` … `</s>`.
2. Probabilitas dengan **add-1 (Laplace) smoothing**:
   - `logP_unigram(w) = log((count(w)+1) / (N+V))`
   - `logP_bigram(prev,w) = log((count(prev,w)+1) / (count(prev)+V))`
3. **Penilaian kandidat (noisy-channel):**
   ```
   score(c) = logP_bigram(prev,c) + logP_bigram(c,next)
              + PRIOR_W · logP_unigram(c)
              − ED_PENALTY · levenshtein(word, c)
   ```
   dengan `ED_PENALTY = 4.0`, `PRIOR_W = 0.3`.
4. `rank_candidates()` mengembalikan **Top-K** kandidat; `best_candidate()`
   mengambil skor tertinggi.

---

## 6. IndoBERT: Pemuatan, Re-ranking & Fine-tuning (Bagian 9–10.B)

- **Pemuatan (Bagian 9):** IndoBERT dimuat sebagai *masked language model*.
  - PRD menyebut `indobenchmark/indobert-base-p1`, namun checkpoint tersebut
    **tidak menyertakan bobot masked-LM head** (prediksi `[MASK]` acak). Karena itu
    dipakai **`indolem/indobert-base-uncased`** yang masked-LM head-nya terlatih.
- **Re-ranking (Bagian 10):** `indobert_logprob(words, idx, cands)` menutup kata
  target dengan `[MASK]`, lalu mengambil **log-prob subword pertama** tiap kandidat
  pada posisi mask → skor kontekstual.
- **Pembagian data (Bagian 10.A):** `typo_dataset.csv` dibagi **80:20**
  (train/validation) memakai `train_test_split`.
- **Fine-tuning (Bagian 10.B):** melatih IndoBERT beberapa **epoch** dengan
  `Dataset`/`DataLoader` PyTorch; **kurva loss train vs validation** diplot sebagai
  bukti proses belajar. Model hasil fine-tuning disimpan di `FINETUNED_DIR`.

> **Catatan GPU:** fine-tuning IndoBERT memerlukan GPU. Komponen klasik
> (kamus + N-Gram) tidak terpengaruh dan dapat dipakai tanpa GPU melalui mode N-Gram.

---

## 7. Pipeline Koreksi Hybrid (Bagian 11)

`correct_text(text, model=...)` adalah antarmuka terpadu untuk ketiga model.
Alur untuk **Hybrid (Model C)**:

1. N-Gram menghasilkan **Top-K** kandidat (`TOPK = 5`).
2. IndoBERT memberi skor kontekstual tiap kandidat.
3. **Skor akhir:** `W_NGRAM·ngram + W_BERT·bert − W_EDIT·editdistance`
   (`W_NGRAM = W_BERT = 1.0`, `W_EDIT = 2.0`).

**Pengaman over-correction** (berlaku untuk SEMUA model):

| Mekanisme | Penjelasan | Param |
|---|---|---|
| Panjang minimum | Kata < 3 huruf (di, ke, ya) dilewati | `MIN_LEN = 3` |
| Proteksi proper noun | Kata berkapital **bukan awal kalimat** tidak dikoreksi | `SKIP_PROPER = True` |
| Lantai frekuensi | Penggantian ke kata sangat langka (`logP_unigram < −12.0`) ditolak | `UNI_FLOOR = −12.0` |

> Dua pengaman terakhir terbukti via `eval_classic.py` + `sweep.py` menaikkan
> Precision jalur klasik **0,53 → 0,86** dan F1 **0,62 → 0,73**, dengan
> over-correction (FP) anjlok ~83%.

---

## 8. Evaluasi (Bagian 12–16)

- **WER & CER (Bagian 12):** `token_edit_distance` untuk WER (level kata) dan CER
  (level karakter), dilaporkan sebagai **reduction** (sebelum vs sesudah koreksi).
- **Harness menyeluruh (Bagian 13–14):** satu kali jalan per model atas 3.000
  pasangan uji, menghitung Accuracy, Precision, Recall, F1, WER/CER reduction, dan
  jumlah over-correction; disajikan sebagai **tabel perbandingan 3 model**.
- **BLEU (Bagian 15):** `corpus_bleu_score` sebagai metrik kualitas kalimat (FR-3).
- **Top-K Accuracy & MRR (Bagian 16):** `rank_candidates_bert` (praseleksi N-Gram
  `pool_k=20` lalu re-rank IndoBERT) menilai posisi kandidat benar dalam ranking;
  target **MRR > 0,80**.

### Hasil jalur klasik N-Gram (3.000 pasangan uji)

| Metrik | Baseline | Final |
|---|---|---|
| Accuracy | 0,914 | **0,953** |
| Precision | 0,546 | **0,856** |
| Recall | 0,760 | 0,635 |
| F1 | 0,635 | **0,729** |
| WER reduction | 0,250 | **0,550** |
| CER reduction | 0,142 | **0,487** |
| Over-correction (FP) | 2.798 | **472** |

> Recall turun karena baseline "tinggi" sebenarnya hasil mengoreksi membabi-buta;
> F1 tetap naik. Mode **Hybrid (IndoBERT)** menjadi pendorong utama Top-1/MRR
> karena memilih kandidat berdasarkan konteks kalimat.

---

## 9. Visualisasi & Error Analysis (Bagian 17–19)

- **VR-1** Confusion Matrix level kata (model C).
- **VR-2** Bar chart metrik klasifikasi 3 model.
- **VR-3** WER & CER sebelum vs sesudah koreksi.
- **VR-4** Top-K Accuracy ranking IndoBERT.
- **VR-5** BLEU (3 model) & MRR.
- **VR-6** Tabel **Error Analysis Dashboard** (model C) sebagai bahan diskusi.
- **Bagian 19** `demo_correction()` — menampilkan kandidat beserta skor bigram &
  skor IndoBERT untuk satu kalimat.

---

## 10. Menyimpan Model untuk Flask (Bagian 20)

Komponen klasik disimpan ke **`models/ngram_model.pkl`** berisi:

- `dictionary`, `unigram`, `bigram`, `V`, `N`.
- `config` — seluruh hyperparameter (`ED_PENALTY`, `PRIOR_W`, `MIN_LEN`, `TOPK`,
  `W_NGRAM`, `W_BERT`, `W_EDIT`, `SKIP_PROPER`, `UNI_FLOOR`) + path `bert_model`
  yang menunjuk ke **IndoBERT hasil fine-tuning** (`FINETUNED_DIR`).

`corrector.py` memuat bundle ini untuk aplikasi Flask (`app.py`), sehingga web app
memakai konfigurasi & model yang identik dengan notebook.

---

## 11. Ringkasan Alur

```
kata salah
   │  is_valid? ──► ya ──► biarkan
   ▼ tidak
candidate generation (edits1 ∪ edits2 ∩ kamus)
   │
   ▼
ranking N-Gram (bigram konteks − penalti edit)  ──► Model B
   │  Top-K
   ▼
re-ranking IndoBERT (skor [MASK] kontekstual)   ──► Model C (Hybrid)
   │
   ▼
pengaman: MIN_LEN · SKIP_PROPER · UNI_FLOOR
   │
   ▼
kata terkoreksi (kapitalisasi & tanda baca dipulihkan)
```
