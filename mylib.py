%tensorflow_version 1.x

  print('Copying Salamander piano SoundFont (via https://sites.google.com/site/soundfonts4u) from GCS...')
  !gsutil -q -m cp -r gs://magentadata/models/music_transformer/primers/* /content/
  !gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /content/

  print('Installing dependencies...')
  !apt-get update -qq && apt-get install -qq libfluidsynth1 build-essential libasound2-dev libjack-dev
  !pip install -q 'tensorflow-datasets < 4.0.0'
  !pip install -qU google-cloud magenta pyfluidsynth

  print('Importing libraries...')

  import numpy as np
  import os
  import tensorflow.compat.v1 as tf

  from google.colab import files

  from tensor2tensor import models
  from tensor2tensor import problems
  from tensor2tensor.data_generators import text_encoder
  from tensor2tensor.utils import decoding
  from tensor2tensor.utils import trainer_lib

  from magenta.models.score2perf import score2perf
  import note_seq

  tf.disable_v2_behavior()

  import ipywidgets as widgets

  from IPython.display import Audio
  from pretty_midi import PrettyMIDI

  #Installeren van niet standaard beschikbare bibliotheken via de PIP tool
  !pip install py_midicsv
  #Importeren van de gebruikte bibliotheken
  import py_midicsv as pm
  import pandas as pd
  from csv import reader
  import csv
  import numpy as np
  pd.set_option("display.max_rows", None, "display.max_columns", None)

  SF2_PATH = '/content/Yamaha-C5-Salamander-JNv5.1.sf2'
SAMPLE_RATE = 16000

# Decode a list of IDs.
def decode(ids, encoder):
  ids = list(ids)
  if text_encoder.EOS_ID in ids:
    ids = ids[:ids.index(text_encoder.EOS_ID)]
  return encoder.decode(ids)