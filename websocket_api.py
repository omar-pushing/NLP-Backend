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
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-this-secret')

# Wildcard CORS - simplest and most reliable
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

usedModel = os.getenv('WHISPER_MODEL', 'tiny')
max_buffer_seconds = int(os.getenv('MAX_BUFFER_SECONDS', 50))
model = whisper.load_model(usedModel)


class RealTimeAudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
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


processor = RealTimeAudioProcessor()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model': usedModel}), 200


@socketio.on('connect')
def handle_connect(auth=None):
    print(f"Client connected: {request.sid}")
    emit('connection_response', {'data': 'Connected'})


@socketio.on('audio_stream')
def handle_audio_stream(data):
    try:
        audio_bytes = bytes(data['audio'])
        processor.add_audio(audio_bytes)
        emit('buffer_update', {'buffer_size': len(processor.audio_buffer)})
    except Exception as e:
        print(f"Error handling audio: {e}")


@socketio.on('transcribe_request')
def handle_transcribe_request():
    try:
        text = processor.transcribe_buffered()
        emit('transcription_result', {
            'text': text if text else 'No audio to transcribe',
            'success': True
        })
    except Exception as e:
        emit('transcription_result', {'text': f'Error: {str(e)}', 'success': False})


@socketio.on('clear_buffer')
def handle_clear_buffer():
    processor.clear_buffer()
    emit('buffer_update', {'buffer_size': 0})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"Starting on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
