from gevent import monkey
monkey.patch_all()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import whisper
import threading
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

# Allow CORS from env-configured frontend URL or all origins
allowed_origins = os.getenv('FRONTEND_URL', '*')

# Apply CORS to all HTTP routes (needed for socket.io polling)
CORS(app, origins=allowed_origins, supports_credentials=True)

socketio = SocketIO(
    app,
    cors_allowed_origins=allowed_origins,
    async_mode='gevent',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
)

usedModel = os.getenv('WHISPER_MODEL', 'tiny')
max_buffer_seconds = int(os.getenv('MAX_BUFFER_SECONDS', 50))

# Load model once
model = whisper.load_model(usedModel)


class RealTimeAudioProcessor:
    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_buffer = deque(maxlen=int(sample_rate * max_buffer_seconds))
        self.lock = threading.Lock()

    def add_audio(self, audio_bytes):
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            with self.lock:
                self.audio_buffer.extend(audio_data)
        except Exception as e:
            print(f"Error adding audio: {e}")

    def get_buffered_audio(self):
        with self.lock:
            if len(self.audio_buffer) > 0:
                return np.array(list(self.audio_buffer), dtype=np.float32)
        return None

    def transcribe_buffered(self):
        try:
            audio = self.get_buffered_audio()
            if audio is None or len(audio) < self.sample_rate:
                return None
            result = model.transcribe(audio, language="en", fp16=False)
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def clear_buffer(self):
        with self.lock:
            self.audio_buffer.clear()


# Global processor
processor = RealTimeAudioProcessor()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model': usedModel}), 200


@socketio.on('connect')
def handle_connect(auth=None):
    print(f"Client connected: {request.sid}")
    emit('connection_response', {'data': 'Connected to real-time API'})


@socketio.on('audio_stream')
def handle_audio_stream(data):
    try:
        audio_bytes = bytes(data['audio'])
        processor.add_audio(audio_bytes)
        emit('buffer_update', {'buffer_size': len(processor.audio_buffer)}, broadcast=False)
    except Exception as e:
        print(f"Error handling audio: {e}")


@socketio.on('transcribe_request')
def handle_transcribe_request():
    try:
        text = processor.transcribe_buffered()
        emit('transcription_result', {
            'text': text if text else 'No audio to transcribe',
            'success': True
        }, broadcast=False)
    except Exception as e:
        emit('transcription_result', {
            'text': f'Error: {str(e)}',
            'success': False
        }, broadcast=False)


@socketio.on('clear_buffer')
def handle_clear_buffer():
    processor.clear_buffer()
    emit('buffer_update', {'buffer_size': 0}, broadcast=False)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting WebSocket Real-Time Speech-to-Text API on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
