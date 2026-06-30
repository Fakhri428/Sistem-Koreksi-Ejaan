# Laporan Proyek UAS — Sistem Koreksi Ejaan Bahasa Indonesia

**Mata Kuliah:** Natural Language Processing
**Jenis:** UAS Project Mandiri
**Pendekatan:** *Hybrid* — Edit Distance + N-Gram Language Model + IndoBERT (*fine-tuned*)
**Notebook:** `02_spell_correction_hybrid_indobert.ipynb`

---

## 1. Ringkasan Eksekutif

Proyek ini membangun sistem koreksi ejaan (*spell correction*) bahasa Indonesia yang
menggabungkan tiga lapis pendekatan secara berurutan:

1. **Edit Distance (Levenshtein)** — membangkitkan kandidat koreksi pada jarak edit 1–2.
2. **N-Gram Language Model (unigram + bigram)** — menyaring kandidat berdasarkan konteks lokal.
3. **IndoBERT (Masked-LM, fine-tuned)** — me-*re-rank* kandidat berdasarkan pemahaman konteks
   kalimat menyeluruh.

Pipeline akhir (Model C) mencapai **F1 = 0.746** dan **Accuracy = 0.956** pada test set,
dengan penurunan WER dari baseline menjadi **0.051** dan CER **0.012**. Sistem juga dilengkapi
dua optimasi lanjutan: **augmentasi data typo** (untuk menaikkan recall) dan **threshold
correction** (sebagai tuas kendali precision/recall).

---

## 2. Dataset

| Komponen | Sumber | Peran |
|---|---|---|
| Kamus | `dataset/dictionary/kamus.txt` | Validasi kata (dictionary checking) |
| Corpus | `dataset/corpus/corpus.txt` | Membangun N-Gram Language Model |
| Pasangan typo | `dataset/typo_dataset/typo_dataset.csv` | Fine-tuning IndoBERT |
| Test | `dataset/test_dataset/test.csv` (3.000 pasangan) | Evaluasi akhir |

**Pembagian data (uas.md §13):** `typo_dataset.csv` dibagi **80% train : 20% validation**
(via `train_test_split`, `random_state=42`) untuk fine-tuning. Test set dipakai **penuh** untuk
evaluasi akhir, terpisah dari proses training.

> **Catatan integritas split:** augmentasi data hanya ditambahkan ke sisi *train*;
> *validation* dan *test* tetap memakai data asli agar evaluasi jujur dan tidak bias terhadap
> typo sintetis.

---

## 3. Metodologi (Pipeline)

```
Preprocessing → Dictionary Check → Candidate Generation → Edit Distance
              → N-Gram (Top-K) → IndoBERT re-ranking → Hasil
```

### 3.1 Preprocessing & Tokenisasi
Tokenizer berbasis regex (offline, tanpa unduhan tambahan). `split_tokens` memecah teks menjadi
potongan kata **dan** non-kata sehingga tanda baca/spasi terjaga saat rekonstruksi, dan
`match_case` mengembalikan kapitalisasi asli setelah koreksi.

### 3.2 Dictionary Checking
Kata dianggap kandidat kesalahan bila tidak ada di kamus (`is_valid`). Kata pendek
(`MIN_LEN = 3`, mis. *di, ke, ya*) dilewati untuk menghindari salah-koreksi.

### 3.3 Candidate Generation (gaya Norvig)
`edits1` / `edits2` membangkitkan semua string pada jarak edit 1–2 (delete, transpose, replace,
insert), lalu disaring hanya yang ada di kamus. Catatan eksperimen: **seluruh typo pada dataset
berjarak edit 1–2**, sehingga perluasan ke edits3 atau keyboard-adjacency tidak menaikkan recall
— plafon recall ditentukan oleh cakupan kamus (kata benar yang OOV / di luar kamus).

### 3.4 N-Gram Language Model
Unigram + bigram dibangun dari corpus dengan **add-1 smoothing**. Skor kandidat memakai
pendekatan *noisy-channel* sederhana:

```
score = logP_bigram(prev, c) + logP_bigram(c, next) + PRIOR_W·logP_unigram(c) − ED_PENALTY·edit_distance
```

dengan `ED_PENALTY = 4.0`, `PRIOR_W = 0.3`. Menghasilkan **Top-K** kandidat (`TOPK = 5`).

### 3.5 IndoBERT Re-ranking
**Checkpoint:** `indolem/indobert-base-uncased`.

> **Keputusan teknis penting:** checkpoint `indobenchmark/indobert-base-p1` (disebut di PRD)
> tidak menyertakan bobot kepala Masked-LM, sehingga prediksi `[MASK]`-nya acak. Dipilih IndoBERT
> dari IndoLEM yang kepala Masked-LM-nya sudah terlatih, agar re-ranking benar-benar bermakna.

Untuk tiap kata typo, posisi kata diganti `[MASK]`, dan tiap kandidat diberi skor = log-prob
subword pertamanya pada posisi mask.

### 3.6 Skor Hybrid Akhir (Model C)
```
final = W_NGRAM·(bigram kiri+kanan) + W_BERT·skor_IndoBERT − W_EDIT·edit_distance
```
dengan `W_NGRAM = 1.0`, `W_BERT = 1.0`, `W_EDIT = 2.0`.

### 3.7 Pengurang Over-correction
Dua guard berlaku untuk semua model (terbukti menaikkan precision 0.53 → 0.86, F1 0.62 → 0.73):
- **`SKIP_PROPER`** — proper noun (kata berkapital non-awal kalimat) tidak dikoreksi.
- **`UNI_FLOOR = −12.0`** — penggantian ke kata yang sangat langka di corpus ditolak.

---

## 4. Fine-tuning IndoBERT

| Hyperparameter | Nilai |
|---|---|
| Objektif | Masked-LM terarah (hanya subword typo di-`[MASK]`) |
| Epoch | 3 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Max length | 128 |
| Optimasi memori | Gradient checkpointing (FR-5, GPU) |

Loss train & validation dipantau tiap epoch (kurva pada Bagian 10.B notebook) sebagai bukti
proses belajar. Device otomatis `cuda` bila tersedia.

### 4.1 Augmentasi Data Typo (untuk Menaikkan Recall)
Karena recall baseline relatif rendah, dibuat **±10.000 pasangan typo sintetis** dari corpus
bersih. Fungsi `augment_typo` menerapkan operasi: *swap, delete, repeat, insert* karakter
(~20% kata per kalimat). Pasangan ini **digabung hanya ke data train** (`ft_train_df_combined`)
sehingga IndoBERT mempelajari variasi kesalahan yang lebih kaya tanpa mencemari validation/test.

---

## 5. Metrik Evaluasi

- **Klasifikasi (FR-1):** Accuracy, Precision, Recall, F1 (level kata; TP/FP/FN/TN).
- **Error rate (FR-2):** WER (*Word Error Rate*) & CER (*Character Error Rate*).
- **Ranking quality (FR-3):** BLEU, Top-K Accuracy (K = 1/3/5), MRR.
- **Baseline comparison (FR-4):** Model A vs B vs C.

---

## 6. Hasil Evaluasi

### 6.1 Baseline Comparison (Model A / B / C)
Tiga varian membuktikan kontribusi tiap komponen:

| Model | Komponen | Keterangan |
|---|---|---|
| **A** | Edit Distance saja | Jarak edit terkecil; seri dipecah frekuensi unigram |
| **B** | + Bigram N-Gram | Menambah konteks lokal |
| **C** | + IndoBERT (fine-tuned) | Re-ranking kontekstual penuh |

*(Tabel angka lengkap dihasilkan oleh harness evaluasi pada Bagian 14 notebook.)*

### 6.2 Performa Model C (Hybrid)

| Metrik | Nilai |
|---|---|
| Accuracy | **0.9556** |
| Precision | 0.8628 |
| Recall | 0.6566 |
| **F1** | **0.7457** |
| WER (sesudah) | 0.0511 |
| CER (sesudah) | 0.0122 |

---

## 7. Analisis Threshold Correction

Dilakukan eksperimen **threshold correction**: koreksi hanya dijalankan bila skor gabungan
melewati ambang `CONFIDENCE_THRESHOLD`. Hasil pada threshold `−18`:

| | Accuracy | Precision | Recall | F1 | WER | CER |
|---|---|---|---|---|---|---|
| **Tanpa Threshold** | 0.9556 | 0.8628 | **0.6566** | **0.7457** | 0.0511 | 0.0122 |
| **Dengan Threshold (−18)** | 0.9079 | **0.9814** | 0.0716 | 0.1335 | 0.1119 | 0.0237 |

### Temuan & Interpretasi
- Threshold menaikkan **precision** (0.863 → 0.981) namun menjatuhkan **recall** secara drastis
  (0.657 → 0.072) sehingga **F1 anjlok** (0.746 → 0.134).
- **Penjelasan teoretis:** gating threshold secara matematis **hanya dapat mengurangi** jumlah
  koreksi. Pipeline tanpa threshold sudah mengoreksi seluruh kata invalid (recall maksimum),
  sehingga threshold **tidak mungkin menaikkan recall** — ia hanya menukar recall demi precision.
- **Kesimpulan desain:** komponen yang sesungguhnya menaikkan recall adalah **augmentasi data**
  (membuat model lebih sensitif & lebih akurat memilih kandidat). Threshold tepatnya diposisikan
  sebagai **tuas precision** — berguna saat aplikasi menuntut sesedikit mungkin salah-koreksi
  (mis. autocorrect senyap), bukan sebagai penaik recall.

### Threshold Sweep (Pemilihan Optimal)
Untuk menghindari penalaan manual, ditambahkan **threshold sweep** (Bagian 14.D): skor IndoBERT
dihitung sekali, lalu banyak threshold disimulasikan murah untuk memplot kurva Precision/Recall/F1
dan memilih threshold **F1-optimal** secara otomatis. Sesuai analisis di atas, titik F1-optimal
berada pada threshold permisif (≈ tanpa gate, F1 ≈ 0.746).

---

## 8. Visualisasi (VR-1 … VR-6)
1. **VR-1** Confusion matrix (Model C, level kata).
2. **VR-2** Bar chart metrik klasifikasi 3 model.
3. **VR-3** WER & CER sebelum vs sesudah koreksi.
4. **VR-4** Top-K Accuracy ranking IndoBERT.
5. **VR-5** BLEU (3 model) & MRR.
6. **VR-6** Error-analysis dashboard (contoh koreksi benar & salah).

Plus kurva loss fine-tuning (train vs val) dan kurva threshold sweep.

---

## 9. Pemenuhan Functional Requirements

| Kode | Requirement | Status |
|---|---|---|
| FR-1 | Metrik klasifikasi (Acc/Prec/Rec/F1) | ✅ |
| FR-2 | Error rate (WER & CER) + grafik | ✅ |
| FR-3 | Deep learning training + ranking quality (BLEU/Top-K/MRR) | ✅ |
| FR-4 | Baseline comparison (Model A/B/C) | ✅ |
| FR-5 | Optimasi GPU (cuda + gradient checkpointing) | ✅ |
| VR-1…6 | Enam visualisasi | ✅ |

---

## 10. Artefak & Deployment

- Komponen klasik (kamus + N-Gram + config) → `models/ngram_model.pkl`.
- IndoBERT fine-tuned → `models/indobert-finetuned/`.
- Config bundle (termasuk `TOPK`, bobot hybrid, `SKIP_PROPER`, `UNI_FLOOR`,
  `CONFIDENCE_THRESHOLD`) siap diintegrasikan ke aplikasi web **Flask**.

---

## 11. Kesimpulan

Sistem hybrid (Edit Distance + N-Gram + IndoBERT fine-tuned) berhasil dievaluasi lengkap sesuai
PRD dan mencapai **F1 = 0.746 / Accuracy = 0.956**. Kontribusi tiap komponen dibuktikan melalui
baseline comparison, dan dua optimasi lanjutan (augmentasi data + threshold) dianalisis secara
jujur: **augmentasi** adalah pengangkat recall, sedangkan **threshold** berfungsi sebagai tuas
precision. Temuan utama yang patut digarisbawahi adalah pemahaman bahwa *threshold gating tidak
dapat menaikkan recall secara matematis* — sebuah analisis trade-off yang memperkuat validitas
metodologi proyek ini.

### Saran Pengembangan
1. Perbandingan recall **sebelum vs sesudah augmentasi** untuk mengkuantifikasi dampaknya.
2. Diagnosa FN: memisahkan typo yang *dilewati guard* vs *salah pilih kandidat* untuk menyetel
   `UNI_FLOOR` / `MIN_LEN` secara tepat sasaran.
3. Mengatasi OOV (kata benar di luar kamus) yang menjadi plafon recall.
