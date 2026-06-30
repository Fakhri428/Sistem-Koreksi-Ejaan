# Product Requirement Document (PRD)

# Sistem Koreksi Ejaan Bahasa Indonesia Berbasis Edit Distance dan N-Gram Language Model

---

## 1. Informasi Proyek

**Nama Proyek**
Sistem Koreksi Ejaan Bahasa Indonesia

**Mata Kuliah**
Natural Language Processing

**Jenis Proyek**
UAS Project Mandiri + Demo Sistem

**Platform**
Web Application (flask)

**Bahasa Pemrograman**
Python 3.8+

---

## 2. Latar Belakang

Dalam penggunaan bahasa Indonesia sehari-hari, sering ditemukan kesalahan penulisan kata akibat kesalahan ketik, kurangnya pemahaman ejaan, atau penggunaan kata yang tidak sesuai.

Kesalahan ejaan dapat menyebabkan teks sulit dipahami dan menurunkan kualitas dokumen digital.

Oleh karena itu, diperlukan sebuah sistem Natural Language Processing (NLP) yang mampu mendeteksi kata yang tidak sesuai dengan kamus bahasa Indonesia dan memberikan rekomendasi perbaikan secara otomatis.

Sistem ini menggunakan pendekatan NLP klasik berupa:

* Edit Distance (Levenshtein Distance)
* Candidate Generation
* N-Gram Language Model

untuk menghasilkan koreksi kata yang paling sesuai berdasarkan kemiripan kata dan konteks kalimat.

---

## 3. Rumusan Masalah

1. Bagaimana mendeteksi kesalahan ejaan pada teks bahasa Indonesia?
2. Bagaimana menghasilkan kandidat kata yang benar dari kata yang salah?
3. Bagaimana menentukan kandidat koreksi terbaik berdasarkan konteks kalimat?
4. Bagaimana membuat sistem koreksi ejaan yang dapat digunakan secara interaktif?

---

## 4. Tujuan Sistem

Membangun aplikasi NLP yang mampu:

* Mendeteksi kesalahan ejaan bahasa Indonesia.
* Memberikan rekomendasi kata yang benar.
* Memilih koreksi terbaik berdasarkan konteks kalimat.
* Menyediakan antarmuka sederhana untuk pengguna.

---

## 5. Target Pengguna

### Pengguna Utama

Mahasiswa, pelajar, dan masyarakat umum yang ingin melakukan pengecekan ejaan teks bahasa Indonesia.

### Contoh Penggunaan

* Memeriksa tugas kuliah.
* Memperbaiki tulisan.
* Membantu penulisan dokumen.

---

## 6. Deskripsi Sistem

Sistem menerima input berupa teks bahasa Indonesia.

Tahapan proses yang dilakukan:

1. Tokenisasi teks.
2. Pemeriksaan kata menggunakan kamus.
3. Deteksi kata yang tidak ditemukan.
4. Pembuatan kandidat koreksi.
5. Perhitungan jarak kata menggunakan Edit Distance.
6. Pemilihan kandidat terbaik menggunakan N-Gram Language Model.
7. Menampilkan hasil koreksi.

---

## 7. Arsitektur Sistem

```text
User Input Text
        |
        v
Text Preprocessing
        |
        v
Word Checking (Kamus Bahasa Indonesia)
        |
   -----------------
   |               |
 Benar           Salah
   |               |
   |               v
   |       Candidate Generation
   |               |
   |               v
   |         Edit Distance
   |               |
   |               v
   |      N-Gram Language Model
   |               |
   -----------------
           |
           v
     Corrected Text
```

---

## 8. Metodologi NLP

### 8.1 Preprocessing

Tahapan:

* Lowercase
* Tokenisasi
* Menghapus karakter yang tidak diperlukan
* Normalisasi teks

**Input**

```text
Saya Suka Makanann Indonesia!!!
```

**Hasil**

```text
saya suka makanann indonesia
```

---

### 8.2 Dictionary Checking

Sistem memiliki kamus kata bahasa Indonesia.

**Kamus**

```text
saya
suka
makan
makanan
indonesia
```

**Input**

```text
saya suka makanann indonesia
```

**Hasil**

```text
makanann = tidak ditemukan
```

---

### 8.3 Candidate Generation

Mencari kata yang memiliki kemiripan dengan kata salah.

**Input**

```text
makanann
```

**Candidate**

```text
makan
makanan
makin
```

---

### 8.4 Edit Distance

Menghitung jumlah perubahan karakter yang diperlukan.

**Contoh**

```text
makanann → makanan
```

Operasi:

* Menghapus satu huruf "n"

Distance:

```text
1
```

Semakin kecil nilai distance, semakin besar kemungkinan kata tersebut benar.

---

### 8.5 N-Gram Language Model

Digunakan untuk memilih kandidat berdasarkan konteks kalimat.

Kalimat:

```text
Saya ingin membeli makanan
```

Probabilitas frasa:

```text
membeli makanan
```

lebih tinggi dibanding:

```text
membeli makin
```

---

## 9. Dataset

### Dataset Kamus

Sumber:

* Kamus Bahasa Indonesia
* Dataset kata bahasa Indonesia publik

Format:

```text
word

saya
makan
makanan
rumah
sekolah
```

Jumlah target:

> Minimal 5000 kata

---

### Dataset Corpus

Digunakan untuk membangun model N-Gram.

Format:

```text
Saya pergi ke sekolah
Saya makan nasi
Saya membeli buku
```

Jumlah target:

> Minimal 500 kalimat

---

## 10. Teknologi yang Digunakan

### Programming

* Python

### Library NLP

* NLTK
* Sastrawi

### Machine Learning

* Scikit-learn

### Interface

* flask

### Data Processing

* Pandas
* NumPy

---

## 11. Struktur Project

```text
spell-checker-nlp/

│
├── dataset/
│   ├── kamus.txt
│   └── corpus.txt
│
├── models/
│   └── ngram_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── edit_distance.py
│   ├── candidate_generator.py
│   ├── language_model.py
│   └── correction.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 12. Fitur Sistem

### Input Text

Pengguna dapat memasukkan teks bahasa Indonesia.

### Deteksi Kesalahan

Contoh:

```text
Kesalahan:
makanann
```

### Rekomendasi Koreksi

```text
makanann

Rekomendasi:

1. makanan
2. makan
```

### Auto Correction

Input:

```text
Saya suka makanann Indonesia
```

Output:

```text
Saya suka makanan Indonesia
```

---

## 13. Evaluasi Sistem

### Pembagian Dataset

```text
Training : 80%
Testing  : 20%
```

### Accuracy

Mengukur ketepatan sistem dalam melakukan koreksi.

Rumus:

```text
Accuracy = Correct Prediction / Total Prediction
```

### Word Error Rate (WER)

Mengukur jumlah kesalahan sebelum dan sesudah proses koreksi.

Target:

Menurunkan jumlah error setelah koreksi dilakukan.

---

## 14. Skenario Pengujian

### Test Case 1

Input:

```text
Saya suka makanann
```

Output:

```text
Saya suka makanan
```

---

### Test Case 2

Input:

```text
Aku pergii ke sekolah
```

Output:

```text
Aku pergi ke sekolah
```

---

## 15. Analisis Error

Kemungkinan kegagalan:

1. Kata baru tidak terdapat dalam kamus.
2. Bahasa slang sulit dikoreksi.
3. Konteks kalimat ambigu.

Contoh:

```text
apel
```

Dapat berarti:

* Buah apel.
* Kegiatan apel.

---

## 16. Output Demo

```text
==========================

Sistem Koreksi Ejaan

Input:
Saya ingin makanann enak

Hasil:
Saya ingin makanan enak

Kata diperbaiki:
makanann -> makanan

==========================
```

---

## 17. Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

* Integrasi IndoBERT.
* Fitur autocomplete.
* Dukungan bahasa daerah.
* Koreksi real-time pada editor teks.

---

## 18. Kesimpulan

Sistem Koreksi Ejaan Bahasa Indonesia merupakan aplikasi NLP yang menerapkan teknik pemrosesan bahasa untuk mendeteksi dan memperbaiki kesalahan penulisan.

Dengan kombinasi Edit Distance dan N-Gram Language Model, sistem mampu menghasilkan koreksi berdasarkan kemiripan kata dan konteks kalimat.
