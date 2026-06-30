"""Aplikasi web Flask untuk Sistem Koreksi Ejaan Bahasa Indonesia.

Menjalankan:
    python app.py
Lalu buka http://127.0.0.1:5000
"""
from flask import Flask, render_template, request, jsonify

from corrector import SpellCorrector

app = Flask(__name__)

# Muat model sekali saat startup (kamus + N-Gram).
print("Memuat model koreksi ejaan ...")
corrector = SpellCorrector()
print("Model siap. Kamus:", len(corrector.dictionary), "kata.")
print("Model IndoBERT (hybrid):", corrector.BERT_NAME)


@app.route("/")
def index():
    return render_template("index.html", bert_name=corrector.BERT_DISPLAY)


@app.route("/api/correct", methods=["POST"])
def api_correct():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    # UI hanya mengekspos mode IndoBERT; N-Gram tetap dipakai internal
    # untuk menghasilkan kandidat Top-K sebelum re-ranking.
    mode = "hybrid"
    if not text:
        return jsonify({"error": "Teks kosong."}), 400
    try:
        result = corrector.correct(text, mode=mode)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Gagal memproses: {exc}"}), 500


if __name__ == "__main__":
    # use_reloader=False: hindari loop reload watchdog di Windows yang dapat
    # memutus koneksi saat IndoBERT sedang dimuat.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
