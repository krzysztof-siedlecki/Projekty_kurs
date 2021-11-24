import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import cv2
import random
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, auc, roc_curve, f1_score, r2_score, confusion_matrix, mean_squared_error
from tensorflow.keras.callbacks import TensorBoard
import datetime
from PIL import Image

st.set_page_config(page_title="WDevs",initial_sidebar_state="expanded")
st.caption('*Klasyfikacja jabłek* :sunglasses: by WDevs, 2021')

st.sidebar.title('Witaj w sortowni jabłek Wdevsów!')
class_names= ['Cortland', 'Gloster', 'Gala', 'Granny Smith', 'Lobo', 'Golden delicious']
file = st.sidebar.file_uploader("Wybierz swoje jabłka", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
if file is not None:
    image = Image.open(file)
    
    try:
      new_model = tf.keras.models.load_model('model.h5')
    except OSError:
      "Nie mogę wczytać modelu..."
    finally:
      img_array = np.array(image)
      img = tf.image.resize(img_array, size=(256,256))
      img = tf.expand_dims(img, axis=0)
      pred = new_model.predict(img)
      st.image(
        image,
        caption=f"Wczytane jabłka: {class_names[np.argmax(pred)]}",
        use_column_width=True,
    )
      
