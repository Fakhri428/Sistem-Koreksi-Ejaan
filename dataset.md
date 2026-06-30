# 9. Dataset

Dataset yang digunakan terdiri dari tiga bagian utama, yaitu dataset kamus, dataset corpus untuk Language Model, dan dataset pasangan kalimat salah-benar untuk Deep Learning.

---

## 9.1 Dataset Kamus Bahasa Indonesia

Dataset kamus digunakan untuk:

* Dictionary Checking
* Candidate Generation
* Perhitungan Edit Distance

### Sumber Dataset

* KBBI (Kamus Besar Bahasa Indonesia)
* Dataset kata Bahasa Indonesia publik
* Kateglo

### Format Dataset

```txt
saya
aku
makan
makanan
pergi
rumah
sekolah
indonesia
```

### Jumlah Target

> Minimal 10.000 kata

### Nama File

```text
dataset/dictionary/kamus.txt
```

---

## 9.2 Dataset Corpus Bahasa Indonesia

Dataset corpus digunakan untuk membangun Bigram dan Trigram Language Model.

### Sumber Dataset

* Indo4B Corpus
* Wikipedia Bahasa Indonesia
* Artikel berita Indonesia
* Dataset teks Bahasa Indonesia publik

### Format Dataset

```txt
Saya pergi ke sekolah.
Saya membeli makanan.
Hari ini cuaca sangat cerah.
Dia membaca buku di perpustakaan.
```

### Jumlah Target

> Minimal 5.000 kalimat

Disarankan:

> 10.000–20.000 kalimat

### Nama File

```text
dataset/corpus/corpus.txt
```

---

## 9.3 Dataset Spell Error untuk Deep Learning

Dataset ini digunakan untuk proses fine-tuning model IndoBERT sehingga mampu memahami dan memperbaiki kesalahan penulisan berdasarkan konteks kalimat.

Dataset terdiri dari pasangan kalimat salah dan kalimat yang benar.

### Sumber Dataset

#### SPECIL (Spell Error Corpus for Indonesian Language)

Merupakan dataset publik yang berisi pasangan kata atau kalimat salah dan benar dalam Bahasa Indonesia.

#### Dataset Typo Generator

Dataset tambahan dibuat secara otomatis dengan menyisipkan kesalahan penulisan dari corpus asli.

Jenis kesalahan yang digunakan:

* Insertion
* Deletion
* Substitution
* Transposition

---

### Format Dataset

```csv
error,correct
Saya suka makanann,Saya suka makanan
Aku pergii ke sekolah,Aku pergi ke sekolah
Dia membelii buku baru,Dia membeli buku baru
```

Contoh lain:

| Error               | Correct            |
| ------------------- | ------------------ |
| Saya makann nasi    | Saya makan nasi    |
| Aku pergii ke rumah | Aku pergi ke rumah |
| Dia membacaa buku   | Dia membaca buku   |

---

### Jumlah Target

Minimal:

> 1.000 pasangan kalimat

Disarankan:

> 5.000 pasangan kalimat

Ideal:

> 10.000–20.000 pasangan kalimat

---

### Pembagian Dataset

#### Training Set

80%

Contoh:

```text
4000 pasangan kalimat
```

Digunakan untuk fine-tuning IndoBERT.

#### Testing Set

20%

Contoh:

```text
1000 pasangan kalimat
```

Digunakan untuk evaluasi model.

---

### Nama File

```text
dataset/typo_dataset/typo_dataset.csv
```

---

## 9.4 Dataset Evaluasi

Dataset evaluasi digunakan untuk mengukur performa sistem.

Metrik yang digunakan:

* Accuracy
* Precision
* Recall
* F1-Score
* Word Error Rate (WER)

### Format Dataset

```csv
error,correct
Saya ingin makanann enak,Saya ingin makanan enak
Aku pergii ke sekolah,Aku pergi ke sekolah
Dia membelii buku,Dia membeli buku
```

### Jumlah Target

> Minimal 500 pasangan kalimat

### Nama File

```text
dataset/test_dataset/test.csv
```

---

## Struktur Dataset

```text
dataset/
│
├── dictionary/
│     └── kamus.txt
│
├── corpus/
│     └── corpus.txt
│
├── typo_dataset/
│     └── typo_dataset.csv
│
└── test_dataset/
      └── test.csv
```

---

## Ringkasan Dataset

| Dataset                                                 | Fungsi                                | Jumlah Ideal                   |
| ------------------------------------------------------- | ------------------------------------- | ------------------------------ |
| **Kamus Bahasa Indonesia**                              | Dictionary Checking dan Edit Distance | 50.000–100.000 kata            |
| **Corpus Bahasa Indonesia (Indo4B, Wikipedia, Berita)** | Bigram dan Trigram Language Model     | 50.000–100.000 kalimat         |
| **SPECIL + Typo Generator**                             | Fine-tuning IndoBERT                  | 20.000–50.000 pasangan kalimat |
| **Test Dataset**                                        | Evaluasi Sistem                       | 2.000–5.000 pasangan kalimat   |

### Detail Dataset

#### 1. Kamus Bahasa Indonesia

Digunakan untuk:

* Dictionary Checking
* Candidate Generation
* Edit Distance

Jumlah ideal:

> 50.000–100.000 kata

Sumber:

* KBBI
* Kateglo
* Daftar kata Bahasa Indonesia publik

---

#### 2. Corpus Bahasa Indonesia

Digunakan untuk:

* Bigram Language Model
* Trigram Language Model

Jumlah ideal:

> 50.000–100.000 kalimat

Sumber:

* Indo4B
* Wikipedia Bahasa Indonesia
* Artikel berita Indonesia
* Dataset SEACrowd

---

#### 3. Dataset Spell Error untuk Deep Learning

Digunakan untuk:

* Fine-tuning IndoBERT
* Contextual Spell Correction

Jumlah ideal:

> 20.000–50.000 pasangan kalimat salah-benar

Sumber:

* SPECIL
* Typo Generator

Jenis typo yang dibangkitkan:

* Insertion
* Deletion
* Substitution
* Transposition

Contoh:

| Error                 | Correct              |
| --------------------- | -------------------- |
| Saya suka makanann    | Saya suka makanan    |
| Aku pergii ke sekolah | Aku pergi ke sekolah |
| Dia membelii buku     | Dia membeli buku     |

---

#### 4. Test Dataset

Digunakan untuk:

* Accuracy
* Precision
* Recall
* F1-Score
* Word Error Rate (WER)

Jumlah ideal:

> 2.000–5.000 pasangan kalimat

Pembagian dataset:

* Training : 80%
* Validation : 10%
* Testing : 10%

Contoh (50.000 pasangan):

* Training : 40.000
* Validation : 5.000
* Testing : 5.000

---

## Kesimpulan

Sistem menggunakan kombinasi dataset kamus, corpus bahasa Indonesia, dan dataset pasangan kalimat salah-benar untuk mendukung pendekatan hybrid yang menggabungkan Edit Distance, N-Gram Language Model, dan Deep Learning menggunakan IndoBERT. Seluruh dataset yang digunakan bersifat gratis dan dapat diperoleh dari sumber publik maupun dibangkitkan secara otomatis menggunakan typo generator.
