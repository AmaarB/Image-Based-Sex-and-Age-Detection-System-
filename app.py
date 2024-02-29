import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
import time


def img_to_display(im):
    #inspried:
    # https://www.kaggle.com/stassl/displaying-inline-images-in-pandas-dataframe
    i = im
    i.thumbnail((500,400), Image.LANCZOS)
    with BytesIO() as buffer:
        i.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def display_result(im, prediction,mode):
    age_icon = "https://cdn-icons-png.flaticon.com/512/3788/3788610.png"
    if prediction[0] <= 0.5:
        sex_icon = "https://i.imgur.com/oAAb8rd.png"
        sex = 'Female'
        prediction[0] = (1-prediction[0])*100

    else:
        sex = "Male"
        sex_icon = "https://i.imgur.com/nxWan2u.png"
        prediction[0] = prediction[0]*100
    
    if prediction[1] <=0.5:
        age = "Old"
        prediction[1] = (1-prediction[1])*100
        age_icon = "https://cdn-icons-png.flaticon.com/512/2512/2512405.png"

    
    else:
        prediction[1] = prediction[1]*100
        age_icon = "https://img.icons8.com/external-flaticons-lineal-color-flat-icons/512/external-youth-football-soccer-flaticons-lineal-color-flat-icons-2.png"
        age = "Young"




# refrenced from https://www.kaggle.com/code/bmarcos/image-recognition-gender-detection-inceptionv3
    display_html = '''
    <div style="overflow: auto;  border: 2px solid #D8D8D8;
        padding: 2px; width: 720px;" >
        <img src="data:image/jpeg;base64,{}" style="float: left;" width="400" height="400">
        <div style="padding: 10px 0px 0px 20px; overflow: auto;">
            <img src="{}" style="float: left;" width="40" height="40">
            <h3 style="margin-left: 70px; margin-top: -5px; font-size: 16px">{}</h3>
            <p style="margin-left: 70px; margin-top: -6px; font-size: 16px">{} prob.</p>
            <p style="margin-left: 70px; margin-top: -16px; font-size: 16px">File Source: {}</p>
            <image src = "{}" style="float: left;" width="40" height="40">
            <h3 style="margin-left: 70px; margin-top: -5px; font-size: 16px">{}</h3>
            <p style="margin-left: 70px; margin-top: -6px; font-size: 16px">{} prob.</p>
            <p style="margin-left: 70px; margin-top: -16px; font-size: 16px">File Source: {}</p>
        </div>
    </div>
    '''.format(img_to_display(im)
               , sex_icon
               , sex
               , "{0:.2f}%".format(prediction[0])
               , mode,age_icon
               , age, "{0:.2f}%".format(prediction[1])
               , mode)
    

    st.markdown(display_html, unsafe_allow_html=True)


def waiting_spinner(waiting_time):
    with st.spinner('processing...'):
        time.sleep(waiting_time)
        st.success('results!')

def sex_and_age_prediction(prediction_scores):
    predictions = []
    for i, label in enumerate(["Male" , "Young"]):
        pred = prediction_scores[i][0][0]
        pred = round(pred,4)


        if label == "Male":
            predictions.append(pred)
        
        else:
            predictions.append(pred)


    return predictions


@st.cache(allow_output_mutation=True)
def load_model(path):
    model = keras.models.load_model((path),
    custom_objects={'KerasLayer':hub.KerasLayer}#https://stackoverflow.com/questions/61814614/unknown-layer-keraslayer-when-i-try-to-load-model
    )
    return model



st.markdown(" <h1 style='text-align: center; '> Age and Sex Predictor</h1>" , unsafe_allow_html=True)

IMAGE_SIZE = (224,224)
path = "model.h5"
url = "https://t3.ftcdn.net/jpg/01/97/11/64/360_F_197116416_hpfTtXSoJMvMqU99n6hGP4xX0ejYa4M7.jpg"
normalization_layer = tf.keras.layers.Rescaling(1. / 255)
#rescales input values to 1/225 becuase og image RGB coefficents is 0-255 which is too high to procces so scole with a 1/255 factor.

model = load_model(path)

mode = st.sidebar.selectbox('''
        Hello there! select the image selection mode!
        please select mode''',
        ['Upload Image',
        'URL'])

if mode == "URL":
    url = st.text_input('Enter URL')
    if url:
        im = Image.open(requests.get(url, stream=True).raw)
        # image = tf.to.decode_image(requests.get(url).content, channels=3, name ="jpeg_reader")
        image = tf.image.resize(im, IMAGE_SIZE)
        image = normalization_layer(image)
        image_holder= st.image(im, caption= 'input image')
        placeholder = st.empty()
        if placeholder.button('Detect The Sex and Age'):
            prediction_scores = model.predict(np.expand_dims(image, axis=0)) # apply prediction using the trained weights
            predictions = sex_and_age_prediction(prediction_scores)
            image_holder.empty()
            display_result(im, predictions, mode)
            placeholder.empty()



elif mode =="Upload Image":
    image_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
    if image_file:
        im = Image.open(image_file)
        image = tf.image.resize(im, IMAGE_SIZE)
        image = normalization_layer(image)
        image_holder = st.image(im, caption = 'Input Image')
        placeholder = st.empty()
        if placeholder.button('Detect The Sex and Age'):
            prediction_scores = model.predict(np.expand_dims(image, axis=0))
            predictions = sex_and_age_prediction(prediction_scores)
            image_holder.empty()
            display_result(im, predictions, mode)
            placeholder.empty()


        
















