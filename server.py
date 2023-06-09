import datetime
from bark import preload_models, SAMPLE_RATE
import nltk
from scipy.io.wavfile import write as write_wav

import os
from os import path
from flask import Flask, request, jsonify
from tts import tts_single, tts_split

preload_models()

#os.environ["SUNO_OFFLOAD_CPU"] = 'True'
#os.environ["SUNO_USE_SMALL_MODELS"] = 'True'
os.environ['GENERATED_AUDIO_DIR'] = path.join(path.dirname(path.realpath(__file__)), 'static/generated')

# Flask app for the tts service


# Set up app
app = Flask(__name__)
app.static_folder = os.environ['GENERATED_AUDIO_DIR']

# Route to serve the static html file
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Route to serve the static wav file
@app.route('/play/<filename>')
def play(filename):
    filepath = path.join(app.static_folder, filename)
    return app.send_static_file(filepath)

# Route to handle tts requests
@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text_prompt = data['text']
    speaker = data.get('speaker', None)
    if data.get('split', False):
        audio_array = tts_split(text_prompt, splitter=nltk.sent_tokenize, speaker=speaker)
    else:
        audio_array = tts_single(text_prompt, speaker=speaker)

    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.wav'
    filepath = path.join(app.static_folder, filename)
    write_wav(filepath, SAMPLE_RATE, audio_array)
    return jsonify({'filename': filename})

if __name__ == '__main__':
    app.run(host='localhost', port=5000)

