"""Harness evaluasi cepat untuk jalur klasik (Edit Distance + N-Gram), tanpa IndoBERT.

Tujuan: mengukur dampak perbaikan candidate generation & ambang skor terhadap
Precision/Recall/F1/WER/CER dan candidate-recall/Top-K secara cepat (CPU only),
sebelum diport ke notebook + corrector.py.

Pakai:
    python eval_classic.py --variant baseline --n 800
    python eval_classic.py --variant improved --n 800
    python eval_classic.py --variant improved --n 0        # 0 = seluruh test set
"""
import argparse
import math
import pickle
import re
import time

import numpy as np
import pandas as pd

BUNDLE = pickle.load(open("models/ngram_model.pkl", "rb"))
dictionary = BUNDLE["dictionary"]
unigram = BUNDLE["unigram"]
bigram = BUNDLE["bigram"]
V = BUNDLE["V"]
N = BUNDLE["N"]

WORD_RE = re.compile(r"[a-zA-Z]+")
SPLIT_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]+")
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# Tetangga keyboard QWERTY (untuk kandidat berbasis salah-pencet tombol).
KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "rygf", "y": "tuhg",
    "u": "yijh", "i": "uokj", "o": "iplk", "p": "ol", "a": "qwsz", "s": "awedxz",
    "d": "serfcx", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "j": "huiknm",
    "k": "jiolm", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def keyboard_edits(word):
    """Kandidat dari salah-pencet tombol bertetangga (subset replaces edits1)."""
    out = set()
    for i, ch in enumerate(word):
        for nb in KEYBOARD_NEIGHBORS.get(ch, ""):
            out.add(word[:i] + nb + word[i + 1:])
    return out


def known(words):
    return {w for w in words if w in dictionary}


def candidates_baseline(word):
    return known([word]) or known(edits1(word)) or known(edits2(word)) or {word}


def candidates_improved(word):
    """Union edits1 + edits2 (tanpa short-circuit) agar koreksi jarak-2 tidak
    hilang ketika ada kandidat jarak-1. Diagnostik membuktikan semua typo pada
    dataset berjarak edit 1-2, sehingga edits3/keyboard tak memberi recall."""
    if word in dictionary:
        return {word}
    pool = known(edits1(word)) | known(edits2(word))
    return pool or {word}


def logP_unigram(w):
    return math.log((unigram[w] + 1) / (N + V))


def logP_bigram(prev, w):
    return math.log((bigram[(prev, w)] + 1) / (unigram[prev] + V))


def is_valid(word):
    return word.lower() in dictionary


# ----------------------- Konfigurasi varian -----------------------
VARIANTS = {
    "baseline": dict(ED_PENALTY=4.0, PRIOR_W=0.3, MIN_LEN=3, W_EDIT=2.0,
                     MARGIN=0.0, candgen="baseline"),
    # Konfigurasi final (terbukti via sweep.py): union candgen + proteksi proper
    # noun + lantai frekuensi. Margin/ED-naik/MIN_LEN-naik diuji & ditolak (lemah).
    "improved": dict(ED_PENALTY=4.0, PRIOR_W=0.3, MIN_LEN=3, W_EDIT=2.0,
                     MARGIN=0.0, candgen="improved",
                     SKIP_PROPER=True, UNI_FLOOR=-12.0),
}


def make_corrector(cfg):
    candgen = candidates_improved if cfg["candgen"] == "improved" else candidates_baseline
    ED_PENALTY = cfg["ED_PENALTY"]
    PRIOR_W = cfg["PRIOR_W"]
    MIN_LEN = cfg["MIN_LEN"]
    MARGIN = cfg["MARGIN"]
    SKIP_PROPER = cfg.get("SKIP_PROPER", False)   # lindungi proper noun (kapital non-awal)
    UNI_FLOOR = cfg.get("UNI_FLOOR", None)        # lantai log-prob unigram kandidat

    def score(c, word, prev_w, next_w):
        lm = logP_bigram(prev_w, c) + logP_bigram(c, next_w)
        return lm + PRIOR_W * logP_unigram(c) - ED_PENALTY * levenshtein(word, c)

    def correct_text(text):
        parts = split_tokens(text)
        word_pos = [i for i, p in enumerate(parts) if p.isalpha()]
        words = [parts[i].lower() for i in word_pos]
        fixed = list(words)
        for k, pos in enumerate(word_pos):
            w = words[k]
            if len(w) < MIN_LEN or is_valid(w):
                continue
            # Proper noun: kata berkapital yang BUKAN awal kalimat -> jangan koreksi.
            if SKIP_PROPER and parts[pos][:1].isupper() and pos != word_pos[0]:
                continue
            prev_w = fixed[k - 1] if k > 0 else "<s>"
            next_w = words[k + 1] if k + 1 < len(words) else "</s>"
            cands = candgen(w)
            best = max(cands, key=lambda c: score(c, w, prev_w, next_w))
            if best == w:
                continue
            # Lantai frekuensi: jangan ganti dengan kata yang sangat langka.
            if UNI_FLOOR is not None and logP_unigram(best) < UNI_FLOOR:
                continue
            # Ambang skor: hanya ganti bila kandidat mengungguli "tetap apa adanya".
            if MARGIN > 0:
                base = score(w, w, prev_w, next_w)
                if score(best, w, prev_w, next_w) - base < MARGIN:
                    continue
            fixed[k] = best
        out = list(parts)
        for k, pos in enumerate(word_pos):
            out[pos] = match_case(parts[pos], fixed[k])
        return "".join(out)

    def rank_candidates(word, prev_w, next_w, k):
        cands = candgen(word)
        return sorted(cands, key=lambda c: score(c, word, prev_w, next_w),
                      reverse=True)[:k]

    return correct_text, rank_candidates, MIN_LEN


# ----------------------- Metrik -----------------------
def token_edit_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def wer(ref, hyp):
    r, h = tokenize(ref), tokenize(hyp)
    return token_edit_distance(r, h) / len(r) if r else 0.0


def cer(ref, hyp):
    r = preprocess(ref).replace(" ", "")
    h = preprocess(hyp).replace(" ", "")
    return levenshtein(r, h) / len(r) if r else 0.0


def evaluate(df, variant, ks=(1, 3, 5)):
    cfg = VARIANTS[variant]
    correct_text, rank_candidates, MIN_LEN = make_corrector(cfg)

    TP = FP = FN = TN = 0
    acc_c = acc_t = 0
    wer_b = wer_a = cer_b = cer_a = 0.0
    nrows = 0
    # Candidate recall / TopK (ranking N-gram) untuk kata typo yang sejajar.
    cand_total = 0
    cand_in_pool = 0
    hits = {k: 0 for k in ks}
    rr = 0.0

    for _, row in df.iterrows():
        err, cor = str(row["error"]), str(row["correct"])
        pred = correct_text(err)
        nrows += 1
        wer_b += wer(cor, err); wer_a += wer(cor, pred)
        cer_b += cer(cor, err); cer_a += cer(cor, pred)

        et, ct, pt = tokenize(err), tokenize(cor), tokenize(pred)
        if len(et) == len(ct) == len(pt):
            acc_t += len(ct)
            acc_c += sum(a == b for a, b in zip(ct, pt))
            for idx, (e, c, p) in enumerate(zip(et, ct, pt)):
                if e != c:
                    TP += int(p == c); FN += int(p != c)
                else:
                    TN += int(p == e); FP += int(p != e)
                # Recall/TopK kandidat (hanya kata yang memang typo & cukup panjang)
                if e != c and len(e) >= MIN_LEN:
                    prev_w = et[idx - 1] if idx > 0 else "<s>"
                    next_w = et[idx + 1] if idx + 1 < len(et) else "</s>"
                    ranked = rank_candidates(e, prev_w, next_w, max(ks))
                    cand_total += 1
                    if c in ranked:
                        cand_in_pool += 1
                        r = ranked.index(c) + 1
                        rr += 1.0 / r
                        for k in ks:
                            hits[k] += int(r <= k)

    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = acc_c / acc_t if acc_t else 0.0
    return {
        "variant": variant, "rows": nrows,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "WER_before": wer_b / nrows, "WER_after": wer_a / nrows,
        "CER_before": cer_b / nrows, "CER_after": cer_a / nrows,
        "WER_reduction": 1 - (wer_a / wer_b) if wer_b else 0.0,
        "CER_reduction": 1 - (cer_a / cer_b) if cer_b else 0.0,
        "cand_recall": cand_in_pool / cand_total if cand_total else 0.0,
        **{f"Top{k}": hits[k] / cand_total if cand_total else 0.0 for k in ks},
        "MRR": rr / cand_total if cand_total else 0.0,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "cand_words": cand_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--n", type=int, default=800, help="0 = full test set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv("dataset/test_dataset/test.csv")
    if args.n and args.n < len(df):
        df = df.sample(args.n, random_state=args.seed).reset_index(drop=True)

    t0 = time.time()
    m = evaluate(df, args.variant)
    dt = time.time() - t0

    print(f"\n=== Varian: {args.variant} | {m['rows']} kalimat | {dt:.1f}s ===")
    order = ["Accuracy", "Precision", "Recall", "F1",
             "WER_before", "WER_after", "WER_reduction",
             "CER_before", "CER_after", "CER_reduction",
             "cand_recall", "Top1", "Top3", "Top5", "MRR"]
    for k in order:
        print(f"  {k:14s}: {m[k]:.4f}")
    print(f"  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']} "
          f"cand_words={m['cand_words']}")


if __name__ == "__main__":
    main()
