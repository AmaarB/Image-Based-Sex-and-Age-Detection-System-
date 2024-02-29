
import pandas as pd
import tensorflow as tf
import pathlib
import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt



df = pd.read_csv("subset.csv")# reading the subset csv file using pandas library
#read subset.csv created from previous file 

IMAGES_PATH = "img_align_celeba" #intializing the image_path with the path of images folder

IMAGE_SIZE = (224, 224) # To later convert the image into to (224, 244) for imagenet, set image_size 224,224


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3) # convert images into tensor to later apply operation it
    image = tf.image.resize(image, IMAGE_SIZE) # resize images to 224,224
    return image
    #array to tensorform, then resize image  

def load_and_preprocess_image(path):
    image = tf.io.read_file(path) # read image and return it as tensor
    return preprocess_image(image)
# read image not resizing just reading


def load_and_preprocess_from_path_label(path, male, young):
    images = load_and_preprocess_image(path) # this funtion will return a processed image with the dimention (224,224,3) rows=224,columns=224, chanels=3 
    return images, male, young



def build_dataset_from_df(df):
    ds = tf.data.Dataset.from_tensor_slices((
        [IMAGES_PATH + image_id for image_id in df["202599"]], # get the actual image path 
        list(df["Male"]), # we will have the second column that contains the sex attribute 
        list(df["Young"])  # column with age attributee 
    ))
    ds = ds.map(load_and_preprocess_from_path_label) #it changes the default dimentions(178×21) of the images to 224x224 to train it with imagenet (previous function to resize image)
    ds = ds.shuffle(buffer_size=1000) # selects the first 1000 images 
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)#it creates a baches of size BATCH_SIZE that will be used later while training the model  
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE) # make the next batch ready for the GPU to apply operation on them
    return ds

    #

df_train = df.sample(frac=0.7) # get the 70% data  of the pandas dataframe df ( line 11)
df_test = df.loc[~df.index.isin(df_train.index)] # get the reaming 30% of the data from the df dataframe 

BATCH_SIZE= 64
train_ds = build_dataset_from_df(df_train) #builing dataset for training
val_ds = build_dataset_from_df(df_test)# building validation dataset

# split data into test and train 

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
preprocessing_model = tf.keras.Sequential([normalization_layer])
# normalisation is used for accuracy and speeding training proceder


def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label
# images are retured in tensor form 

#train_ds = train_ds.map(process)
train_ds = train_ds.map(lambda images, male, young:
                        (preprocessing_model(images), (male, young))) #precocess the train dataset

# we are using map function to train dataset  and apply into the fucntion we have useing preproccesing model (research online)


val_ds = val_ds.map(lambda images, male, young:
                    (normalization_layer(images), (male, young)))# preprocess the validation datadset

# we are using map function to train dataset and apply into the fucntion we have useing normalization model (research online)

import tensorflow_hub as hub #TensorFlow Hub is a repository of trained machine learning models ready for fine-tuning and deployable anywhere. Reuse trained models like
do_fine_tuning = False
# concept of tine tuning pre trained 
MODEL_HANDLE = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2" #Feature vectors of images with EfficientNet V2 with input size 224x224, trained on imagenet-21k
input = tf.keras.Input(shape=IMAGE_SIZE + (3,)) # define the input layer of a Keras model, shape of our image, image zie (224,224) 3 dimension
x = hub.KerasLayer(MODEL_HANDLE, trainable=do_fine_tuning)(input) 
# donwloading the pretained 
x = tf.keras.layers.Dropout(rate=0.2)(x)
# dropout (research)
x = tf.keras.layers.Dense(128, activation="relu")(x)
# dense (reshearch), activeatip = relu, 

out_male = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation="sigmoid", name='male')(x)
out_young = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation="sigmoid", name='young')(x)
# output, one for male and one for young 


model = tf.keras.Model(inputs = input, outputs = [out_male, out_young])
print(model.summary())

# two attributes for keras model, one as input and one as outputs (out_male, out_young)

#compiling the model
model.compile(
    loss = {
        "male": tf.keras.losses.BinaryCrossentropy(),#define the lossfunction
        "young": tf.keras.losses.BinaryCrossentropy()##define the lossfunction
    },
    metrics = {
        "male": 'accuracy',
        "young": 'accuracy'
    },
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)# defining the optimizer
)

steps_per_epoch = len(df_train) // BATCH_SIZE #define the steps per epoch on training
validation_steps = len(df_test) // BATCH_SIZE #defining the steps per epcoh on validation
callbacks = [keras.callbacks.ModelCheckpoint("model.h5")] # save a model or weights (in a checkpoint file) at some interval
hist = model.fit(
    train_ds,
    epochs=3, steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks).history

#ploting the loss and accuracy graphgs
fig, ax = plt.subplots(2, 2, figsize=(15, 12))
for i, c in enumerate(["male", "young"]):
    ax[i, 0].plot(hist[f"{c}_loss"], label="train")
    ax[i, 0].plot(hist[f"val_{c}_loss"], label="val")
    ax[i, 0].set_title(f"Loss ({c})")
    ax[i, 0].legend()
    ax[i, 1].plot(hist[f"{c}_accuracy"], label="train")
    ax[i, 1].plot(hist[f"val_{c}_accuracy"], label="val")
    ax[i, 1].set_title(f"Accuracy ({c})")
    ax[i, 1].legend()
plt.show()

#perform the prediction
x, y = next(iter(val_ds)) #get the element of the dataset, each element contains the images and label that are eqaul to the batch size 
image = x[0, :, :, :]# get the first image from the list of images
plt.imshow(image)# show the image
plt.axis('off')
plt.show()

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

path = "model.h5"
def load_model(path):

  model = keras.models.load_model(
                                        (path),
        custom_objects={'KerasLayer':hub.KerasLayer}#https://stackoverflow.com/questions/61814614/unknown-layer-keraslayer-when-i-try-to-load-model
  )
  return model
model = load_model(path)

