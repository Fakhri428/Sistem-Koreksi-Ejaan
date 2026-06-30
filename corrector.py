"""Mesin koreksi ejaan bahasa Indonesia.

Memuat komponen yang dihasilkan notebook (`models/ngram_model.pkl`):
kamus + N-Gram Language Model. Menyediakan dua mode koreksi:

* `mode="ngram"`   : Edit Distance + N-Gram Language Model (cepat, tanpa GPU).
* `mode="hybrid"`  : N-Gram (Top-K) + re-ranking IndoBERT (lebih akurat, lebih berat).

IndoBERT dimuat secara lazy (hanya saat mode hybrid pertama kali dipakai).
"""
import os
import re
import math
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ngram_model.pkl")
FINETUNED_DIR = os.path.join(BASE_DIR, "models", "indobert-finetuned")
DEFAULT_BERT = "indolem/indobert-base-uncased"


def resolve_bert_name(raw):
    """Mengembalikan referensi model IndoBERT yang siap dipakai.

    `raw` bisa berupa path lokal (mis. ``models\\indobert-finetuned`` yang
    tersimpan di pkl dengan gaya Windows) atau ID Hugging Face Hub. Path lokal
    dinormalisasi dan dijadikan absolut terhadap folder proyek agar tidak
    bergantung pada direktori kerja saat aplikasi dijalankan.
    """
    if raw:
        candidate = raw.replace("\\", os.sep).replace("/", os.sep)
        local = candidate if os.path.isabs(candidate) else os.path.join(BASE_DIR, candidate)
        if os.path.isdir(local):
            return local
        # Jika tampak seperti path lokal (ada pemisah) tapi tak ditemukan,
        # jangan paksakan ke Hub — coba model fine-tuned bawaan.
        if (os.sep in candidate or "/" in raw or "\\" in raw) and os.path.isdir(FINETUNED_DIR):
            return FINETUNED_DIR
        if os.sep not in candidate:
            return raw  # anggap sebagai ID Hugging Face Hub
    # Tanpa konfigurasi: utamakan model fine-tuned lokal bila tersedia.
    return FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else DEFAULT_BERT

WORD_RE = re.compile(r"[a-zA-Z]+")
SPLIT_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]+")
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def tokenize(text):
    return WORD_RE.findall(text.lower())


def split_tokens(text):
    return SPLIT_RE.findall(text)


def match_case(original, corrected):
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected.capitalize()
    return corrected


def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET]
    inserts = [L + c + R for L, R in splits for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


class SpellCorrector:
    def __init__(self, model_path=MODEL_PATH):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.dictionary = bundle["dictionary"]
        self.unigram = bundle["unigram"]
        self.bigram = bundle["bigram"]
        self.V = bundle["V"]
        self.N = bundle["N"]
        cfg = bundle.get("config", {})
        self.ED_PENALTY = cfg.get("ED_PENALTY", 4.0)
        self.PRIOR_W = cfg.get("PRIOR_W", 0.3)
        self.MIN_LEN = cfg.get("MIN_LEN", 3)
        self.TOPK = cfg.get("TOPK", 5)
        self.W_NGRAM = cfg.get("W_NGRAM", 1.0)
        self.W_BERT = cfg.get("W_BERT", 1.0)
        self.W_EDIT = cfg.get("W_EDIT", 2.0)
        # Pengurang over-correction (lihat eval_classic.py / sweep.py):
        #   SKIP_PROPER -> jangan koreksi proper noun (kata berkapital non-awal).
        #   UNI_FLOOR   -> jangan ganti dengan kata yang sangat langka di korpus
        #                  (log-prob unigram di bawah ambang). -12.0 ~= frekuensi <10.
        self.SKIP_PROPER = cfg.get("SKIP_PROPER", True)
        self.UNI_FLOOR = cfg.get("UNI_FLOOR", -12.0)
        self.BERT_NAME = resolve_bert_name(cfg.get("bert_model", ""))
        # Nama ringkas untuk ditampilkan di UI (path lokal -> nama folder).
        if os.path.isdir(self.BERT_NAME):
            self.BERT_DISPLAY = os.path.basename(self.BERT_NAME.rstrip(os.sep)) + " (fine-tuned)"
        else:
            self.BERT_DISPLAY = self.BERT_NAME

        # Komponen IndoBERT (lazy)
        self._bert_tokenizer = None
        self._bert_model = None
        self._device = None

    # ---------- Dictionary & candidate generation ----------
    def is_valid(self, word):
        return word.lower() in self.dictionary

    def known(self, words):
        return {w for w in words if w in self.dictionary}

    def candidates(self, word):
        if word in self.dictionary:
            return {word}
        # Union edits1 + edits2 agar koreksi jarak-2 tidak hilang saat ada
        # kandidat jarak-1 (short-circuit lama menjatuhkan recall jarak-2).
        pool = self.known(edits1(word)) | self.known(edits2(word))
        return pool or {word}

    # ---------- N-Gram language model ----------
    def logP_unigram(self, w):
        return math.log((self.unigram[w] + 1) / (self.N + self.V))

    def logP_bigram(self, prev, w):
        return math.log((self.bigram[(prev, w)] + 1) / (self.unigram[prev] + self.V))

    def score_candidate(self, c, word, prev_w, next_w):
        lm = self.logP_bigram(prev_w, c) + self.logP_bigram(c, next_w)
        prior = self.logP_unigram(c)
        ed = levenshtein(word, c)
        return lm + self.PRIOR_W * prior - self.ED_PENALTY * ed

    def rank_candidates(self, word, prev_w="<s>", next_w="</s>", k=3):
        cands = self.candidates(word)
        return sorted(cands,
                      key=lambda c: self.score_candidate(c, word, prev_w, next_w),
                      reverse=True)[:k]

    def best_candidate(self, word, prev_w="<s>", next_w="</s>"):
        cands = self.candidates(word)
        return max(cands, key=lambda c: self.score_candidate(c, word, prev_w, next_w))

    # ---------- IndoBERT (lazy) ----------
    def _ensure_bert(self):
        if self._bert_model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM, logging
        logging.set_verbosity_error()
        self._torch = torch
        self._bert_tokenizer = AutoTokenizer.from_pretrained(self.BERT_NAME)
        self._bert_model = AutoModelForMaskedLM.from_pretrained(self.BERT_NAME)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._bert_model.to(self._device).eval()

    def indobert_logprob(self, words, idx, cands):
        torch = self._torch
        masked = list(words)
        masked[idx] = self._bert_tokenizer.mask_token
        text = " ".join(masked)
        enc = self._bert_tokenizer(text, return_tensors="pt").to(self._device)
        mask_pos = (enc["input_ids"][0] == self._bert_tokenizer.mask_token_id).nonzero()
        if len(mask_pos) == 0:
            return {c: 0.0 for c in cands}
        mask_pos = mask_pos[0].item()
        with torch.no_grad():
            logits = self._bert_model(**enc).logits[0, mask_pos]
        logprobs = torch.log_softmax(logits, dim=-1)
        scores = {}
        for c in cands:
            ids = self._bert_tokenizer(c, add_special_tokens=False)["input_ids"]
            scores[c] = logprobs[ids[0]].item() if ids else -1e9
        return scores

    def best_candidate_hybrid(self, word, words, idx):
        prev_w = words[idx - 1] if idx > 0 else "<s>"
        next_w = words[idx + 1] if idx + 1 < len(words) else "</s>"
        cands = self.rank_candidates(word, prev_w, next_w, k=self.TOPK)
        if len(cands) == 1:
            return cands[0]
        bert = self.indobert_logprob(words, idx, cands)

        def final(c):
            ngram = self.logP_bigram(prev_w, c) + self.logP_bigram(c, next_w)
            return (self.W_NGRAM * ngram + self.W_BERT * bert[c]
                    - self.W_EDIT * levenshtein(word, c))

        return max(cands, key=final)

    # ---------- Public API ----------
    def correct(self, text, mode="ngram"):
        """Mengoreksi teks. Mengembalikan dict berisi hasil lengkap untuk UI."""
        if mode == "hybrid":
            self._ensure_bert()

        parts = split_tokens(text)
        word_pos = [i for i, p in enumerate(parts) if p.isalpha()]
        words = [parts[i].lower() for i in word_pos]
        fixed = list(words)
        corrections = []
        first_pos = word_pos[0] if word_pos else -1

        for k, pos in enumerate(word_pos):
            w = words[k]
            if len(w) < self.MIN_LEN or self.is_valid(w):
                continue
            # Proper noun: kata berkapital yang BUKAN awal kalimat -> biarkan.
            if self.SKIP_PROPER and parts[pos][:1].isupper() and pos != first_pos:
                continue
            if mode == "hybrid":
                cand = self.best_candidate_hybrid(w, fixed, k)
            else:
                prev_w = fixed[k - 1] if k > 0 else "<s>"
                next_w = words[k + 1] if k + 1 < len(words) else "</s>"
                cand = self.best_candidate(w, prev_w, next_w)
            # Lantai frekuensi: tolak penggantian ke kata yang sangat langka.
            if cand != w and self.logP_unigram(cand) < self.UNI_FLOOR:
                continue
            fixed[k] = cand
            if cand != w:
                corrections.append({"from": w, "to": cand})

        # Rekonstruksi teks keluaran (jaga tanda baca & kapitalisasi)
        out_parts = list(parts)
        changed_positions = set()
        for k, pos in enumerate(word_pos):
            out_parts[pos] = match_case(parts[pos], fixed[k])
            if fixed[k] != words[k]:
                changed_positions.add(pos)

        output = "".join(out_parts)
        output_html = self._build_html(out_parts, changed_positions)

        return {
            "input": text,
            "output": output,
            "output_html": output_html,
            "corrections": corrections,
            "num_errors": len(corrections),
            "mode": mode,
        }

    @staticmethod
    def _build_html(out_parts, changed_positions):
        from html import escape
        chunks = []
        for i, p in enumerate(out_parts):
            esc = escape(p)
            if i in changed_positions:
                chunks.append(f'<mark class="fixed">{esc}</mark>')
            else:
                chunks.append(esc)
        return "".join(chunks)
