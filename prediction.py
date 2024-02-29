import tensorflow_hub as hub 
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os
import numpy as np
from tensorflow import keras


IMAGE_SIZE = (224,224)
path = "model.h5"

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
url = "https://media.distractify.com/brand-img/3JWAIT4PU/0x0/jamale-meme-tiktok-1666816782403.jpg"

def load_model(path):

  model = keras.models.load_model((path),
        custom_objects={'KerasLayer':hub.KerasLayer}#https://stackoverflow.com/questions/61814614/unknown-layer-keraslayer-when-i-try-to-load-model
  )
  return model


model = load_model(path)
normalization_layer = tf.keras.layers.Rescaling(1. / 255)
image = tf.io.decode_image(requests.get(url).content, channels=3, name="jpeg_reader")

plt.imshow(image)# show the image
plt.axis('off')
plt.show()

image = tf.image.resize(image, IMAGE_SIZE)
image = normalization_layer(image)

prediction_scores = model.predict(np.expand_dims(image, axis=0))# apply prediction using the trained weights


for i, label in enumerate(["Male", "Young"]):
    pred = prediction_scores[i][0][0]
    
    if label == "Male":
      if pred > 0.5:
        print("predicted as Male :", pred)
      else:
        print("predicted as Female :", 1-pred)
        
    else:    
      if pred > 0.5:
        print("predicted as Young :", pred)
      else:
        print("predicted as Old:", 1-pred)

## code comment are on end of modeltraining.py file given that its mostly the same code.