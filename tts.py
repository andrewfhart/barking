from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import nltk
import numpy as np

preload_models()

text_prompt = "Hello, world!"

SPLIT=False

if (SPLIT):
  SPEAKER="v2/en_speaker_9"
  sentences = nltk.sent_tokenize(text_prompt)
  silence = np.zeros(int(0.25*SAMPLE_RATE))
  pieces = []
  for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]
  write_wav("latest.wav", SAMPLE_RATE, np.concatenate(pieces))
else:
  audio_array = generate_audio(text_prompt)
  write_wav("latest.wav", SAMPLE_RATE, audio_array)

print("Done.")
