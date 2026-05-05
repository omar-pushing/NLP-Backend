"""
VoiceFlow Backend — HTTP REST API
POST /transcribe  →  receives raw WAV/PCM audio, returns transcribed text
GET  /health      →  liveness check
"""
import os
import io
import tempfile
import traceback

import numpy as np
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)

WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'tiny')
print(f"Loading Whisper '{WHISPER_MODEL}'…")
model = whisper.load_model(WHISPER_MODEL)
print(f"Whisper '{WHISPER_MODEL}' ready.")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': WHISPER_MODEL})


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Accepts audio in two ways:
      1. multipart/form-data  with field 'audio' (blob/file, any format ffmpeg understands)
      2. application/octet-stream  raw body = 32-bit float PCM at 16 kHz mono

    Returns JSON:
      { "text": "...", "success": true }
    or on error:
      { "text": "", "success": false, "error": "..." }
    """
    try:
        content_type = request.content_type or ''

        if 'multipart/form-data' in content_type:
            # ── WAV/WebM/OGG blob from MediaRecorder ──────────────────────────
            if 'audio' not in request.files:
                return jsonify({'text': '', 'success': False,
                                'error': 'No audio field in form data'}), 400

            audio_file = request.files['audio']
            suffix = _ext_from_mime(audio_file.mimetype)

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                audio_file.save(tmp.name)
                tmp_path = tmp.name

            try:
                result = model.transcribe(tmp_path, fp16=False)
            finally:
                os.unlink(tmp_path)

        elif 'application/octet-stream' in content_type:
            # ── Raw float32 PCM @ 16 kHz mono ─────────────────────────────────
            raw = request.get_data()
            if not raw:
                return jsonify({'text': '', 'success': False,
                                'error': 'Empty body'}), 400
            audio = np.frombuffer(raw, dtype=np.float32)
            result = model.transcribe(audio, language='en', fp16=False)

        else:
            return jsonify({'text': '', 'success': False,
                            'error': f'Unsupported content-type: {content_type}'}), 415

        text = (result.get('text') or '').strip()
        print(f"[transcribe] => {text[:100]!r}")
        return jsonify({'text': text, 'success': True})

    except Exception:
        traceback.print_exc()
        return jsonify({'text': '', 'success': False,
                        'error': 'Internal server error'}), 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ext_from_mime(mime: str) -> str:
    mime = (mime or '').lower()
    if 'webm'  in mime: return '.webm'
    if 'ogg'   in mime: return '.ogg'
    if 'mp4'   in mime: return '.mp4'
    if 'mpeg'  in mime: return '.mp3'
    if 'wav'   in mime: return '.wav'
    return '.webm'   # safe default — ffmpeg handles it


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"Dev server on :{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
