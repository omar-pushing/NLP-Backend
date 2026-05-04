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
CHUNK_SECONDS = int(os.getenv('CHUNK_SECONDS', 5))   # transcribe every N seconds of audio
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS           # 80 000 samples @ 5 s
MIN_SAMPLES   = SAMPLE_RATE * 1                       # need at least 1 s before attempting

usedModel = os.getenv('WHISPER_MODEL', 'tiny')
model = whisper.load_model(usedModel)
print(f"Whisper model '{usedModel}' loaded.")


# ── Per-session state ─────────────────────────────────────────────────────────

class SessionProcessor:
    """One instance per connected socket client."""

    def __init__(self, sid: str):
        self.sid = sid
        self.pending: list = []          # raw float32 samples not yet transcribed
        self.lock = threading.Lock()
        self._busy = False               # True while Whisper is running
        self._flush_requested = False    # True when stop was pressed mid-transcription

    # ── audio ingestion ───────────────────────────────────────────────────────

    def add_audio(self, payload) -> int:
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

    # ── transcription helpers ─────────────────────────────────────────────────

    def _run_whisper(self, audio_np: np.ndarray) -> str:
        try:
            result = model.transcribe(audio_np, language="en", fp16=False)
            return (result.get("text") or "").strip()
        except Exception as e:
            print(f"[{self.sid}] Whisper error: {e}")
            return ""

    def _make_emit(self, sid):
        def _emit(event, data):
            socketio.emit(event, data, to=sid)
        return _emit

    # ── chunk worker (runs in background thread) ──────────────────────────────

    def _worker(self, emit_fn, is_final: bool = False):
        """
        Grab a chunk (or everything on final flush), transcribe, emit.
        Loops until pending is exhausted if is_final, otherwise does one chunk.
        """
        while True:
            with self.lock:
                if is_final:
                    if len(self.pending) < MIN_SAMPLES:
                        self.pending.clear()
                        self._busy = False
                        emit_fn('transcription_result', {'text': '', 'success': True, 'final': True})
                        return
                    chunk = list(self.pending)
                    self.pending.clear()
                else:
                    if len(self.pending) < CHUNK_SAMPLES:
                        self._busy = False
                        return
                    chunk = self.pending[:CHUNK_SAMPLES]
                    self.pending = self.pending[CHUNK_SAMPLES:]

            audio_np = np.array(chunk, dtype=np.float32)
            text = self._run_whisper(audio_np)

            if text:
                print(f"[{self.sid}] {'final' if is_final else 'chunk'} → '{text[:80]}'")
                emit_fn('transcription_result', {
                    'text': text,
                    'success': True,
                    'final': is_final,
                })

            if is_final:
                with self.lock:
                    self._busy = False
                emit_fn('transcription_result', {'text': '', 'success': True, 'final': True})
                return

            # For non-final: check if there's another full chunk ready
            with self.lock:
                if len(self.pending) >= CHUNK_SAMPLES:
                    continue   # loop again — process next chunk immediately
                self._busy = False
                return

    def maybe_transcribe(self):
        """Called after every audio_stream event. Kicks off worker if needed."""
        with self.lock:
            if self._busy or len(self.pending) < CHUNK_SAMPLES:
                return
            self._busy = True
            sid = self.sid

        emit_fn = self._make_emit(sid)
        t = threading.Thread(target=self._worker, args=(emit_fn, False), daemon=True)
        t.start()

    def flush(self):
        """Called when the user stops recording. Transcribes whatever remains."""
        with self.lock:
            if self._busy:
                # Worker is running; mark that a flush is needed — it will be
                # handled after the current chunk finishes via _flush_after_busy
                self._flush_requested = True
                return
            if len(self.pending) < MIN_SAMPLES:
                self.pending.clear()
                socketio.emit('transcription_result', {'text': '', 'success': True, 'final': True}, to=self.sid)
                return
            self._busy = True
            sid = self.sid

        emit_fn = self._make_emit(sid)
        t = threading.Thread(target=self._worker, args=(emit_fn, True), daemon=True)
        t.start()

    def clear(self):
        with self.lock:
            self.pending.clear()
            self._flush_requested = False


# ── Session registry ──────────────────────────────────────────────────────────

sessions: dict = {}
sessions_lock = threading.Lock()


def get_session(sid: str) -> SessionProcessor:
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = SessionProcessor(sid)
        return sessions[sid]


def remove_session(sid: str):
    with sessions_lock:
        sessions.pop(sid, None)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model': usedModel, 'chunk_seconds': CHUNK_SECONDS}), 200


# ── Socket.IO events ──────────────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect(auth=None):
    sid = request.sid
    get_session(sid)
    print(f"Client connected: {sid}")
    emit('connection_response', {'data': 'Connected'})


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    remove_session(sid)
    print(f"Client disconnected: {sid}")


@socketio.on('audio_stream')
def handle_audio_stream(data):
    sid = request.sid
    session = get_session(sid)
    buf_len = session.add_audio(data.get('audio', b''))
    emit('buffer_update', {'buffer_size': buf_len})
    # Auto-transcribe once we've collected a full chunk
    session.maybe_transcribe()


@socketio.on('stop_recording')
def handle_stop_recording():
    """Client fires this the moment the user clicks Stop."""
    sid = request.sid
    get_session(sid).flush()


@socketio.on('transcribe_request')
def handle_transcribe_request():
    """Legacy / manual trigger — same as stop_recording."""
    sid = request.sid
    get_session(sid).flush()


@socketio.on('clear_buffer')
def handle_clear_buffer():
    sid = request.sid
    get_session(sid).clear()
    emit('buffer_update', {'buffer_size': 0})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"Starting VoiceFlow backend on port {port} …")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
