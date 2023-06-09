from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
import nltk
import numpy as np

def tts_single(text_prompt, speaker = None):
  history_prompt = speaker if speaker else None
  audio_array = generate_audio(text_prompt, history_prompt=history_prompt)
  return audio_array

def tts_split(text_prompt, splitter = nltk.sent_tokenize, speaker = None):
  sentences = splitter(text_prompt)
  silence = np.zeros(int(0.25*SAMPLE_RATE))
  pieces = []
  history_prompt = speaker if speaker else None
  for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=history_prompt)
    pieces += [audio_array, silence.copy()]
  return np.concatenate(pieces)