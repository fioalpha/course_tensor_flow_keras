import streamlit as st
import tensorflow as tf
import io 
from PIL import Image
import gdown
import numpy as np
import pandas as pd
import plotly.express as px



@st.cache
def carrega_model(): 
    # https://drive.google.com/file/d/1Q4fp9RRP1JY7kdGKbSfUPQT5VaJ_VCTo/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1Q4fp9RRP1JY7kdGKbSfUPQT5VaJ_VCTo'


    name_model = "modelo_quantizado16bit.tflite"
    gdown.download(url, name_model)
    interpreter = tf.lite.Interceptor(model_path=name_model)
    interpreter.allocate_tensors()

    return interpreter

def carrega_image(): 
    unperloaded_file = st.file_uploader('Arraster e solte', type=['png',' jpg', 'jpeg'])

    if unperloaded_file is not None: 
        image_data = unperloaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.success("Imagem foi carregado com sucesso")

        image = np.array(image,  dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index', image])
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes=['BlackMeasles', 'BlackRot', 'LeafBlight', 'HealthyGrapes']

    df = pd.Dataframe()
    df['classed'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(
        df, 
        y='Classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title='Probabilidade de Classes de Doenças em Uvas'
    )

    st.plotly_chart(fig)
def main(): 
    st.set_page_config(
        page_title='Classificação Folhas de Videira'
    )
    st.write("# Classificação folhas de Videiras! ")

    interperter=carrega_model()

    image=carrega_image()

    if image is not None: 
        previsao(interperter, image)


if __name__ == '__main__':
    main()
