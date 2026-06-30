# Product Requirement Document (PRD)

# Sistem Koreksi Ejaan Bahasa Indonesia Berbasis Edit Distance, N-Gram Language Model, dan IndoBERT

---

# 1. Informasi Proyek

**Nama Proyek**

Sistem Koreksi Ejaan Bahasa Indonesia Berbasis Hybrid NLP dan Deep Learning

**Mata Kuliah**

Natural Language Processing

**Jenis Proyek**

UAS Project Mandiri + Demo Sistem

**Platform**

Web Application (Flask)

**Bahasa Pemrograman**

Python 3.10+

---

# 2. Latar Belakang

Kesalahan penulisan (spelling error) sering terjadi dalam dokumen digital, baik akibat kesalahan ketik maupun kurangnya pemahaman terhadap ejaan bahasa Indonesia yang benar.

Kesalahan tersebut dapat menurunkan kualitas informasi dan mempengaruhi pemahaman pembaca.

Untuk mengatasi masalah tersebut, diperlukan sistem yang mampu mendeteksi dan memperbaiki kesalahan ejaan secara otomatis berdasarkan kemiripan kata dan konteks kalimat.

Sistem ini menggabungkan:

* Edit Distance (Levenshtein Distance)
* Candidate Generation
* N-Gram Language Model
* Deep Learning menggunakan IndoBERT

Pendekatan hybrid ini diharapkan menghasilkan koreksi yang lebih akurat dibanding metode konvensional.

---

# 3. Rumusan Masalah

1. Bagaimana mendeteksi kata yang salah pada teks bahasa Indonesia?
2. Bagaimana menghasilkan kandidat koreksi dari kata yang salah?
3. Bagaimana memilih kandidat terbaik berdasarkan konteks kalimat?
4. Bagaimana memanfaatkan Deep Learning untuk meningkatkan akurasi koreksi?
5. Bagaimana membangun aplikasi koreksi ejaan berbasis web menggunakan Flask?

---

# 4. Tujuan Sistem

Membangun sistem yang mampu:

* Mendeteksi kesalahan ejaan.
* Menghasilkan kandidat koreksi.
* Memilih kata terbaik menggunakan Language Model.
* Memanfaatkan IndoBERT untuk memahami konteks kalimat.
* Menampilkan hasil koreksi melalui aplikasi web.

---

# 5. Target Pengguna

## Pengguna Utama

* Mahasiswa
* Pelajar
* Penulis
* Pengguna umum

## Contoh Penggunaan

* Memperbaiki tugas kuliah.
* Memeriksa artikel.
* Membantu penulisan laporan.
* Koreksi teks bahasa Indonesia.

---

# 6. Deskripsi Sistem

Tahapan sistem:

1. Input teks dari pengguna.
2. Preprocessing.
3. Tokenisasi.
4. Dictionary Checking.
5. Candidate Generation.
6. Perhitungan Edit Distance.
7. Ranking menggunakan N-Gram Language Model.
8. Pemilihan Top-K kandidat.
9. Re-ranking menggunakan IndoBERT.
10. Menampilkan hasil koreksi.

---

# 7. Arsitektur Sistem

```text
User Input
     |
     v
Preprocessing
     |
     v
Dictionary Checking
     |
     v
Candidate Generation
     |
     v
Edit Distance
     |
     v
N-Gram Language Model
     |
     v
Top-K Candidate
     |
     v
IndoBERT
(Contextual Ranking)
     |
     v
Corrected Text
```

---

# 8. Metodologi NLP

## 8.1 Preprocessing

Tahapan:

* Lowercase
* Tokenization
* Menghapus karakter khusus
* Normalisasi teks

Input:

```text
Saya Sukaa Makanann!!!
```

Output:

```text
saya sukaa makanann
```

---

## 8.2 Dictionary Checking

Kata yang tidak ditemukan dalam kamus dianggap sebagai kandidat kesalahan.

Contoh:

```text
makanann
```

Status:

```text
Tidak ditemukan
```

---

## 8.3 Candidate Generation

Menghasilkan kandidat berdasarkan kemiripan kata.

Input:

```text
makanann
```

Candidate:

```text
makanan
makan
makin
```

---

## 8.4 Edit Distance

Menghitung jarak antar kata.

Contoh:

```text
makanann -> makanan
```

Distance:

```text
1
```

---

## 8.5 N-Gram Language Model

Memilih kandidat berdasarkan probabilitas konteks.

Contoh:

Kalimat:

```text
Saya ingin makanann enak
```

Bigram:

```text
ingin makanan
```

memiliki probabilitas lebih tinggi dibanding:

```text
ingin makan
```

---

## 8.6 Deep Learning Menggunakan IndoBERT

Model memahami konteks kalimat secara keseluruhan.

Input:

```text
Saya ingin [MASK] enak
```

Kandidat:

* makanan
* makan

Probabilitas:

| Kata    | Probabilitas |
| ------- | ------------ |
| makanan | 0.92         |
| makan   | 0.08         |

Output:

```text
Saya ingin makanan enak
```

---

# 9. Dataset

## Dataset Kamus

Sumber:

* KBBI
* Dataset kata bahasa Indonesia publik

Jumlah:

> Minimal 5.000 kata

---

## Dataset Corpus

Digunakan untuk membangun N-Gram.

Target:

> Minimal 500 kalimat

---

## Dataset Deep Learning

Berisi pasangan kalimat salah dan benar.

Contoh:

| Input Salah           | Target Benar         |
| --------------------- | -------------------- |
| Saya suka makanann    | Saya suka makanan    |
| Aku pergii ke sekolah | Aku pergi ke sekolah |

Jumlah target:

> Minimal 1.000 pasangan kalimat

---

# 10. Teknologi yang Digunakan

## Programming

* Python

## NLP

* NLTK
* Sastrawi

## Deep Learning

* PyTorch
* Transformers
* IndoBERT

Model:

```text
indobenchmark/indobert-base-p1
```

## Machine Learning

* Scikit-learn

## Web Framework

* Flask

## Data Processing

* Pandas
* NumPy

---

# 11. Struktur Project

```text
spell-correction-system/

│
├── dataset/
│     ├── kamus.txt
│     ├── corpus.txt
│     └── typo_dataset.csv
│
├── models/
│     ├── ngram_model.pkl
│     └── indobert_model/
│
├── src/
│     ├── preprocessing.py
│     ├── dictionary_checker.py
│     ├── candidate_generator.py
│     ├── edit_distance.py
│     ├── ngram_model.py
│     ├── indobert_ranker.py
│     └── correction.py
│
├── templates/
│     └── index.html
│
├── static/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

# 12. Fitur Sistem

## Input Text

Pengguna memasukkan teks.

---

## Deteksi Kesalahan

Menampilkan kata yang salah.

---

## Candidate Suggestion

Menampilkan rekomendasi kata.

---

## Auto Correction

Menghasilkan kalimat yang telah diperbaiki.

---

## Highlight Error

Menandai kata yang salah.

---

## Menampilkan Kata yang Dikoreksi

Contoh:

```text
makanann → makanan
pergii → pergi
```

---

# 13. Evaluasi Sistem

## Pembagian Dataset

```text
Training 80%
Testing 20%
```

---

## Accuracy

Mengukur ketepatan prediksi.

---

## Precision

Mengukur kualitas prediksi positif.

---

## Recall

Mengukur kemampuan menemukan koreksi yang benar.

---

## F1 Score

Menggabungkan precision dan recall.

---

## Word Error Rate (WER)

Mengukur jumlah kesalahan sebelum dan sesudah koreksi.

---

# 14. Skenario Pengujian

## Test Case 1

Input:

```text
Saya suka makanann
```

Output:

```text
Saya suka makanan
```

---

## Test Case 2

Input:

```text
Aku pergii ke sekolah
```

Output:

```text
Aku pergi ke sekolah
```

---

# 15. Analisis Error

Kemungkinan kegagalan:

1. Kata baru belum terdapat dalam kamus.
2. Bahasa gaul sulit dipahami model.
3. Konteks kalimat ambigu.
4. Nama orang atau nama tempat dapat dianggap salah.

---

# 16. Tampilan Sistem

```text
===================================

Sistem Koreksi Ejaan Bahasa Indonesia

Input:

Saya ingin makanann enak

Output:

Saya ingin makanan enak

Kata yang diperbaiki:

makanann → makanan

===================================
```

---

# 17. Pengembangan Selanjutnya

* Fine-tuning IndoBERT khusus typo bahasa Indonesia.
* Koreksi real-time.
* REST API.
* Multi-user.
* Export PDF.
* Dukungan bahasa daerah.
* Integrasi editor teks.

---

# 18. Kesimpulan

Sistem Koreksi Ejaan Bahasa Indonesia merupakan sistem hybrid yang menggabungkan Edit Distance, N-Gram Language Model, dan Deep Learning menggunakan IndoBERT.

Pendekatan ini memungkinkan sistem menghasilkan koreksi yang lebih akurat karena mempertimbangkan kemiripan kata serta konteks kalimat secara menyeluruh.

Framework Flask digunakan untuk menyediakan antarmuka berbasis web yang ringan dan mudah digunakan.
