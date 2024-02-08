import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import tensorflow as tf
import cv2
import math

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from struct import unpack
# from tqdm import tqdm

st.header("Image View Classifier")
prod_cat = st.selectbox(label="Select Product Category", options=["Bags","Clothes","Schuhe","Waesche"])
prod_cat_class = st.selectbox(label="Select Product View", options=[""])
file = st.file_uploader("Please choose a file")

if file is not None:
    img_data = file.getvalue()
    st.image(img_data, caption='Model Prediction = {}')