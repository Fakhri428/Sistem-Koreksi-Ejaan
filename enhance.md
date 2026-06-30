# Product Requirement Document (PRD)

# Improvement Plan — Sistem Koreksi Ejaan Bahasa Indonesia Berbasis Hybrid Edit Distance, N-Gram Language Model, dan IndoBERT

---

# 1. Informasi Proyek

## Nama Proyek

Sistem Koreksi Ejaan Bahasa Indonesia Berbasis Hybrid Edit Distance, Bigram Language Model, dan IndoBERT

## Mata Kuliah

Natural Language Processing

## Tujuan Perbaikan

Meningkatkan kualitas evaluasi, visualisasi hasil eksperimen, serta optimasi inferensi IndoBERT menggunakan PyTorch dan GPU agar sistem memenuhi standar akademik NLP dan siap dipresentasikan pada UAS.

---

# 2. Kondisi Saat Ini

Pipeline saat ini:

Input Teks

↓

Tokenisasi

↓

Edit Distance Candidate Generation

↓

Bigram Language Model Scoring

↓

Top-K Candidate Selection

↓

IndoBERT Re-ranking

↓

Output Koreksi

Kelebihan:

* Sudah menggunakan hybrid approach
* Memanfaatkan konteks melalui Bigram
* Memanfaatkan semantic understanding melalui IndoBERT
* Dataset relatif besar

Kekurangan:

* Evaluasi belum lengkap
* Visualisasi minim
* Belum memanfaatkan GPU secara optimal
* Belum ada ranking metrics untuk IndoBERT
* Belum ada analisis kandidat Top-K

---

# 3. Target Perbaikan

## Functional Requirements

### FR-1 Evaluasi Klasik NLP

Tambahkan metrik:

* Accuracy
* Precision
* Recall
* F1-Score

Tujuan:

Mengukur kemampuan sistem mendeteksi dan memperbaiki typo secara keseluruhan.

---

### FR-2 Error Rate Evaluation

Tambahkan:

* Word Error Rate (WER)
* Character Error Rate (CER)

Tujuan:

Mengukur penurunan kesalahan sebelum dan sesudah koreksi.

Formula:

WER

WER = (S + D + I) / N

CER

CER = (Sc + Dc + Ic) / Nc

---

### FR-3 Deep Learning Evaluation

Karena menggunakan IndoBERT, tambahkan:

#### BLEU Score

Mengukur kemiripan hasil koreksi dengan ground truth.

Library:

nltk.translate.bleu_score

---

#### Top-K Accuracy

Evaluasi:

Top-1 Accuracy
Top-3 Accuracy
Top-5 Accuracy

Pertanyaan yang dijawab:

"Apakah jawaban benar muncul di kandidat terbaik?"

---

#### Mean Reciprocal Rank (MRR)

Mengukur posisi kandidat benar dalam ranking IndoBERT.

Formula:

MRR = (1/N) Σ (1/rank)

Semakin tinggi semakin baik.

Target:

MRR > 0.8

---

### FR-4 Baseline Comparison

Bandingkan:

Model A

Edit Distance

Model B

Edit Distance + Bigram

Model C

Edit Distance + Bigram + IndoBERT

Output:

Tabel perbandingan metrik.

Tujuan:

Membuktikan kontribusi IndoBERT.

---

# 4. Optimasi GPU

## FR-5 Migrasi ke PyTorch GPU

Deteksi perangkat:

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Load model:

```python
model.to(device)
```

Inference:

```python
inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
```

Target:

* Inference lebih cepat
* Presentasi demo lebih lancar

---

# 5. Visualisasi

## VR-1 Confusion Matrix

Visualisasi:

* TP
* TN
* FP
* FN

Library:

```python
sklearn.metrics.confusion_matrix
seaborn
```

---

## VR-2 Evaluation Metrics Bar Chart

Menampilkan:

* Accuracy
* Precision
* Recall
* F1

Tujuan:

Visualisasi performa model.

---

## VR-3 WER dan CER Comparison

Grafik:

Sebelum Koreksi vs Sesudah Koreksi

Metrik:

* WER
* CER

Tujuan:

Menunjukkan efektivitas sistem.

---

## VR-4 Top-K Accuracy Chart

Menampilkan:

* Top-1
* Top-3
* Top-5

Tujuan:

Menunjukkan kualitas ranking IndoBERT.

---

## VR-5 BLEU dan MRR Visualization

Bar chart:

BLEU
MRR

Untuk menunjukkan kualitas semantic ranking.

---

## VR-6 Error Analysis Dashboard

Contoh tabel:

| Input  | Prediksi | Ground Truth | Status  |
| ------ | -------- | ------------ | ------- |
| sayaa  | saya     | saya         | Correct |
| makn   | makan    | makan        | Correct |
| ikna   | ikan     | ikan         | Correct |
| berjai | berjalan | bekerja      | Wrong   |

Tujuan:

Bahan diskusi pada presentasi.

---

# 6. Eksperimen

## Eksperimen 1

Edit Distance Only

Evaluasi:

* Accuracy
* Precision
* Recall
* F1
* WER
* CER

---

## Eksperimen 2

Edit Distance + Bigram

Evaluasi sama.

---

## Eksperimen 3

Edit Distance + Bigram + IndoBERT

Evaluasi:

* Accuracy
* Precision
* Recall
* F1
* WER
* CER
* BLEU
* Top-K Accuracy
* MRR

---

# 7. Deliverables

## Notebook Final

File:

spell_correction_hybrid_final.ipynb

---

## Visualisasi

* Confusion Matrix
* Accuracy Chart
* WER/CER Chart
* BLEU Chart
* Top-K Accuracy Chart
* MRR Chart

---

## Demo

Input:

"Sya mkan nasi dan ikna"

Output:

"Saya makan nasi dan ikan"

Disertai:

* kandidat koreksi
* skor bigram
* skor IndoBERT

---

# 8. Target Hasil

| Metric         | Target |
| -------------- | ------ |
| Accuracy       | > 90%  |
| Precision      | > 70%  |
| Recall         | > 75%  |
| F1-Score       | > 72%  |
| WER Reduction  | > 30%  |
| CER Reduction  | > 35%  |
| BLEU           | > 0.80 |
| Top-3 Accuracy | > 95%  |
| Top-5 Accuracy | > 98%  |
| MRR            | > 0.80 |

---

# Kesimpulan

Perbaikan difokuskan pada:

1. Evaluasi NLP yang lebih lengkap.
2. Evaluasi khusus Deep Learning.
3. Baseline comparison.
4. Optimasi GPU menggunakan PyTorch.
5. Visualisasi untuk kebutuhan laporan dan presentasi UAS.

Metode utama tetap sesuai topik Sistem Koreksi Ejaan karena Edit Distance dan N-Gram tetap menjadi inti sistem, sedangkan IndoBERT digunakan sebagai tahap re-ranking untuk meningkatkan kualitas koreksi berbasis konteks.
