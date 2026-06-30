# Sistem Koreksi Ejaan Bahasa Indonesia
### Pendekatan Hybrid: Edit Distance + N-Gram Language Model + IndoBERT (Fine-Tuned)

Nama: Fakhri
NIM: [isi NIM]
Mata Kuliah: Natural Language Processing — UAS
Program Studi Informatika, Universitas Harapan Bangsa
Tanggal: [isi tanggal presentasi]

---

# Latar Belakang & Motivasi

Kesalahan ketik (typo) adalah masalah umum pada teks Bahasa Indonesia yang menurunkan kualitas pemrosesan teks lanjutan: pencarian informasi, analisis sentimen, klasifikasi dokumen, hingga chatbot.

**Masalah yang ingin diselesaikan:**
- Pendekatan klasik (Edit Distance, N-Gram) cepat tapi kurang memahami konteks kalimat
- Model Transformer (IndoBERT) memahami konteks lebih baik, tapi mahal jika dipakai sendirian untuk seluruh proses
- Dibutuhkan pipeline **hybrid** yang menggabungkan kecepatan pendekatan klasik dengan pemahaman kontekstual deep learning

**Tujuan proyek:**
- Membangun dataset Bahasa Indonesia mandiri & reproducible
- Membangun pipeline koreksi ejaan 3 lapis (Edit Distance → N-Gram → IndoBERT)
- Fine-tuning IndoBERT untuk re-ranking kandidat
- Mengevaluasi sistem secara menyeluruh & membandingkan kontribusi tiap komponen

---

# Dataset

Empat dataset dibangun mandiri dengan SEED = 42 (reproducible), kombinasi sumber publik terverifikasi + fallback generator offline.

| Dataset | Fungsi | Sumber Utama | Jumlah |
|---|---|---|---|
| Kamus | Dictionary checking & edit distance | geovedi/indonesian-wordlist + Hunspell id_ID | 100.000 kata |
| Korpus | Bigram & trigram language model | Leipzig Corpora Collection (ind_news_2022_100K) | 96.715 kalimat |
| Typo dataset | Fine-tuning IndoBERT | Diturunkan dari korpus (kolam train) | 30.000 pasangan |
| Test dataset | Evaluasi akhir | Diturunkan dari korpus (kolam test, disjoint) | 3.000 pasangan |

**Preprocessing:**
- Kamus: lowercase, filter regex huruf saja, top-up afiksasi (prefix/suffix sah) hingga 100.000 kata
- Korpus: filter panjang kalimat 3–30 kata & rasio huruf >60%
- Typo pairs: 4 operasi karakter — insertion, deletion, substitution (keyboard QWERTY), transposition
- **Anti data leakage:** korpus dipisah disjoint sebelum dibuat pasangan train (82.208 kalimat) & test (14.507 kalimat)

---

# Arsitektur & Metodologi

**Pipeline 5 tahap, dijalankan berurutan per kata:**

1. **Preprocessing & Tokenisasi** — regex tokenizer, jaga tanda baca & kapitalisasi
2. **Dictionary Checking** — kata dianggap typo bila tidak ada di kamus (kata < 3 huruf dilewati)
3. **Candidate Generation** (gaya Norvig) — bangkitkan semua kata jarak edit 1–2, saring yang ada di kamus
4. **N-Gram Language Model** — unigram + bigram (add-1 smoothing), noisy-channel scoring, hasilkan Top-5 kandidat
5. **IndoBERT Re-ranking** — skor log-prob token [MASK] dari IndoBERT fine-tuned, dikombinasi dengan skor N-Gram

**Skor akhir (Model C):**
`final = W_NGRAM·(bigram) + W_BERT·(IndoBERT) − W_EDIT·(jarak edit)`
W_NGRAM=1.0, W_BERT=1.0, W_EDIT=2.0

**Guard anti over-correction:**
- SKIP_PROPER → nama diri/kata berkapital tidak dikoreksi
- UNI_FLOOR = −12.0 → tolak kandidat sangat langka di korpus

**Checkpoint model:** `indolem/indobert-base-uncased` (dipilih karena kepala Masked-LM-nya sudah terlatih, beda dengan checkpoint `indobenchmark/indobert-base-p1` yang prediksinya acak)

---

# Implementasi & Tools

**Pipeline alur:**
```
Preprocessing → Dictionary Check → Candidate Generation (Edit Distance)
            → N-Gram (Top-K) → IndoBERT Re-ranking → Hasil Koreksi
```

**Fine-tuning IndoBERT:**

| Hyperparameter | Nilai |
|---|---|
| Objektif | Masked-LM terarah (hanya subword typo di-mask) |
| Epoch | 3 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Max length | 128 |
| Optimasi | Gradient checkpointing (GPU) |

**Augmentasi data:** ±10.000 pasangan typo sintetis (swap/delete/repeat/insert) — hanya masuk ke data train, validation & test tetap data asli

**Library & tools utama:** Python, PyTorch, Hugging Face Transformers, pandas, NumPy, scikit-learn, NLTK (BLEU), Jupyter Notebook

---

# Hasil Evaluasi

**Baseline comparison — kontribusi tiap komponen:**

| Model | Komponen |
|---|---|
| A | Edit Distance saja |
| B | + Bigram N-Gram |
| C | + IndoBERT (fine-tuned) |

**Performa Model C (pipeline hybrid penuh) — 3.000 pasangan test:**

| Metrik | Nilai |
|---|---|
| Accuracy | 0,9556 |
| Precision | 0,8628 |
| Recall | 0,6566 |
| **F1-Score** | **0,7457** |
| WER (sesudah) | 0,0511 |
| CER (sesudah) | 0,0122 |

**Dampak threshold correction:**

| Skenario | Precision | Recall | F1 |
|---|---|---|---|
| Tanpa threshold | 0,8628 | 0,6566 | **0,7457** |
| Dengan threshold (−18) | 0,9814 | 0,0716 | 0,1335 |

Threshold menaikkan precision drastis tapi menjatuhkan recall & F1 → bukan solusi optimal di sini.

---

# Analisis & Diskusi (Error Analysis)

**Kapan sistem berhasil:**
- Typo berjarak edit 1–2 dengan konteks kalimat jelas → IndoBERT efektif memilih kandidat yang sesuai
- Kata umum yang sering muncul di korpus → skor N-Gram & unigram mendukung pilihan yang tepat

**Kapan sistem gagal:**
1. **Plafon recall akibat OOV** — kata benar yang tidak ada di kamus (±8,8%) tidak pernah dikenali, sehingga recall maksimum dibatasi cakupan kamus, bukan oleh kekuatan model
2. **Ambiguitas kandidat berjarak sama** — beberapa kandidat punya jarak edit sama & konteks kalimat pendek, menyebabkan salah pilih kandidat
3. **Trade-off precision vs recall pada threshold** — threshold gating secara matematis hanya bisa mengurangi jumlah koreksi, sehingga tidak pernah menaikkan recall, hanya menukar recall demi precision

**Faktor yang terbukti efektif:**
- Guard SKIP_PROPER & UNI_FLOOR → menaikkan precision 0,53 → 0,86, F1 0,62 → 0,73
- Augmentasi data typo sintetis → pengangkat recall sesungguhnya (bukan threshold)

**Catatan metodologis:** kategori False Negative menggabungkan "tidak terdeteksi" dan "salah pilih kandidat" — perlu dipisah pada pengembangan berikutnya untuk diagnosis lebih presisi

---

# Kesimpulan

- Pipeline hybrid (Edit Distance + N-Gram + IndoBERT fine-tuned) berhasil dibangun & dievaluasi lengkap: **F1 = 0,746, Accuracy = 0,956**
- Baseline comparison membuktikan tiap komponen (N-Gram, IndoBERT) menambah kontribusi positif dibanding Edit Distance saja
- Augmentasi data = pengangkat recall; threshold correction = tuas precision (bukan sebaliknya)

**Keterbatasan:**
- Recall masih dibatasi cakupan kamus (kata benar OOV tidak pernah terdeteksi)
- Evaluasi memakai typo sintetis, belum diuji pada typo nyata dari pengguna
- Definisi Precision/Recall belum memisahkan "salah pilih kandidat" sebagai kategori tersendiri

**Rencana pengembangan:**
1. Perbandingan recall sebelum vs sesudah augmentasi secara eksplisit
2. Diagnosis FN lebih granular (guard vs salah pilih kandidat)
3. Perluasan cakupan kamus / deteksi kata valid berbasis korpus
4. Pengujian pada data typo nyata pengguna
5. Integrasi ke aplikasi web (Flask) sebagai demo

---

# Demo Preview

**Artefak yang siap diintegrasikan:**
- `models/ngram_model.pkl` — komponen klasik (kamus + N-Gram + config)
- `models/indobert-finetuned/` — IndoBERT hasil fine-tuning
- Config bundle: TOPK, bobot hybrid, SKIP_PROPER, UNI_FLOOR, CONFIDENCE_THRESHOLD

**Rencana demo:** aplikasi web sederhana (Flask) — pengguna mengetik kalimat dengan typo, sistem menampilkan kalimat hasil koreksi beserta kata yang diubah

[Tempel screenshot/demo aplikasi di sini]

**Terima kasih**
Pertanyaan & Diskusi
