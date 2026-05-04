from gevent import monkey
monkey.patch_all()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import whisper
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-this-secret')

CORS(app, origins="*")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

SAMPLE_RATE   = 16000
CHUNK_SECONDS = int(os.getenv('CHUNK_SECONDS', 5))
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS   # 80 000 samples = 5 s
MIN_SAMPLES   = SAMPLE_RATE * 1               # at least 1 s

usedModel = os.getenv('WHISPER_MODEL', 'tiny')
model = whisper.load_model(usedModel)
print(f"Whisper '{usedModel}' loaded.")


class SessionProcessor:
    def __init__(self, sid):
        self.sid     = sid
        self.pending = []            # float32 samples
        self.lock    = threading.Lock()
        self._worker_thread = None   # currently running thread, or None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _emit(self, event, data):
        socketio.emit(event, data, to=self.sid)

    def _run_whisper(self, samples):
        try:
            audio = np.array(samples, dtype=np.float32)
            result = model.transcribe(audio, language="en", fp16=False)
            return (result.get("text") or "").strip()
        except Exception as e:
            print(f"[{self.sid}] Whisper error: {e}")
            return ""

    # ── audio ingestion ───────────────────────────────────────────────────────

    def add_audio(self, payload):
        try:
            if isinstance(payload, (bytes, bytearray)):
                raw = payload
            elif isinstance(payload, list):
                raw = bytes(payload)
            else:
                raw = bytes(payload)
            samples = np.frombuffer(raw, dtype=np.float32)
            with self.lock:
                self.pending.extend(samples.tolist())
                return len(self.pending)
        except Exception as e:
            print(f"[{self.sid}] add_audio error: {e}")
            return 0

    # ── worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        """
        Runs in a background thread.
        Drains self.pending in CHUNK_SAMPLES-sized bites.
        Exits only when pending is empty, then emits final=True.
        Because it holds no lock while Whisper runs, a concurrent add_audio
        can safely extend self.pending — we'll pick it up on the next loop.
        """
        while True:
            # ── grab next chunk ──────────────────────────────────────────────
            with self.lock:
                n = len(self.pending)

                if n == 0:
                    # Nothing left — we're done
                    self._worker_thread = None
                    self._emit('transcription_result', {'text': '', 'success': True, 'final': True})
                    return

                if n >= CHUNK_SAMPLES:
                    chunk = self.pending[:CHUNK_SAMPLES]
                    self.pending = self.pending[CHUNK_SAMPLES:]
                    is_final = False
                else:
                    # Less than a full chunk — this only happens during a flush
                    # (the worker is started by flush() when n >= MIN_SAMPLES)
                    if n < MIN_SAMPLES:
                        self.pending.clear()
                        self._worker_thread = None
                        self._emit('transcription_result', {'text': '', 'success': True, 'final': True})
                        return
                    chunk = list(self.pending)
                    self.pending.clear()
                    is_final = True

            # ── transcribe (lock NOT held) ───────────────────────────────────
            text = self._run_whisper(chunk)
            if text:
                print(f"[{self.sid}] → '{text[:80]}'")

            self._emit('transcription_result', {
                'text': text or '',
                'success': True,
                'final': is_final,
            })

            if is_final:
                with self.lock:
                    self._worker_thread = None
                return
            # else loop — check if more samples accumulated while we were busy

    # ── public API ────────────────────────────────────────────────────────────

    def maybe_transcribe(self):
        """Start worker if a full chunk is ready and no worker is running."""
        with self.lock:
            if self._worker_thread is not None:
                return   # already running — it will loop and pick up more data
            if len(self.pending) < CHUNK_SAMPLES:
                return
            t = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread = t
        t.start()

    def flush(self):
        """
        Called on Stop. If a worker is already running it will naturally drain
        all pending samples and emit final=True when done. If not, start one.
        """
        with self.lock:
            if self._worker_thread is not None:
                # Worker is running and will keep looping until pending is empty,
                # then emit final=True by itself — nothing to do here.
                return
            if len(self.pending) < MIN_SAMPLES:
                # Nothing worth transcribing
                self.pending.clear()
                # Emit final directly (no thread needed)
                sid = self.sid
            else:
                t = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread = t
                sid = None

        if sid:
            socketio.emit('transcription_result', {'text': '', 'success': True, 'final': True}, to=sid)
        else:
            t.start()

    def clear(self):
        with self.lock:
            self.pending.clear()
            # Don't kill the worker — it will emit final=True on its own when
            # it finds pending empty, which is fine.


# ── Session registry ──────────────────────────────────────────────────────────

sessions = {}
sessions_lock = threading.Lock()

def get_session(sid):
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = SessionProcessor(sid)
        return sessions[sid]

def remove_session(sid):
    with sessions_lock:
        sessions.pop(sid, None)


# ── Routes & events ───────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': usedModel, 'chunk_seconds': CHUNK_SECONDS})

@socketio.on('connect')
def on_connect(auth=None):
    get_session(request.sid)
    print(f"+ {request.sid}")
    emit('connection_response', {'data': 'Connected'})

@socketio.on('disconnect')
def on_disconnect():
    remove_session(request.sid)
    print(f"- {request.sid}")

@socketio.on('audio_stream')
def on_audio(data):
    s = get_session(request.sid)
    n = s.add_audio(data.get('audio', b''))
    emit('buffer_update', {'buffer_size': n})
    s.maybe_transcribe()

@socketio.on('stop_recording')
def on_stop():
    get_session(request.sid).flush()

@socketio.on('transcribe_request')   # legacy
def on_transcribe():
    get_session(request.sid).flush()

@socketio.on('clear_buffer')
def on_clear():
    get_session(request.sid).clear()
    emit('buffer_update', {'buffer_size': 0})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"Starting on :{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
