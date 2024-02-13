import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import gradcam as gcam # Helper file contains the class definition for GradCAM
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
# from tf_explain.core.grad_cam import GradCAM

# @st.cache_data    
def load_models():
    models = {}
    model_bags = load_model('models/bag_restnet50_model_ft_l5_93%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_clothes = load_model('models/clothes_restnet50_func_model_97%', custom_objects={'imagenet_utils': imagenet_utils})
    model_schuhe = load_model('models/schuhe_model_resnet_ft.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_waesche = load_model('models/waesch_funcResnet_model_94%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    models['Bags'] = model_bags
    models['Clothes'] = model_clothes
    models['Schuhe'] = model_schuhe
    models['Waesche'] = model_waesche
    return models

def image_processing_function(im_path, input_img_dims, pre_process_function=None):

    orig = image.load_img(im_path)
    orig_arr = image.img_to_array(orig).astype("uint8")
    img = image.load_img(im_path, target_size=input_img_dims)

    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)

    return img, image_arr, orig_arr

models = load_models()
# explainer = GradCAM()

model = ""

view_labels = {
    "Bags": ["Front","Side","Inside","Back","Look"],
    "Clothes": ["Model Front","Zoomed","Model Back","Ghost","Look"],
    "Schuhe": ["Overll to Right","Back","Top or Sole","Side to Left","Zoom"],
    "Waesche": ["Model Front","Zoomed","Model Back","Ghost","Look"]
}

st.set_page_config(page_title="Image View Classifier", layout="wide")
st.header("Image View Classifier")
prod_cat = st.selectbox(label="Select Product Category", options=["Bags","Clothes","Schuhe","Waesche"])
prod_cat_class = st.selectbox(label="Select Product View", options=view_labels.get(prod_cat))
model = models.get(prod_cat)

uploaded_image = st.file_uploader("Please choose an image...", type=["jpg", "jpeg", "png"])

container = st.container(border=True)
left_column1, middle_column1, right_column1 = st.columns([1,1,1])

# start_over()

if uploaded_image is not None:
    # img_data = file.getvalue()
    # img = Image.open(uploaded_image)
    # img = image.load_img(uploaded_image)
    # img = image.img_to_array(img)
    # img = image.load_img(uploaded_image, target_size=(224, 224))
    input_img_dims = (427, 350)
    img, img_arr, orig = image_processing_function(uploaded_image, input_img_dims)
    # predictions = model.predict(np.expand_dims(img, axis=0))
    predictions = model.predict(img_arr)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    predicted_label = view_labels.get(prod_cat)[predicted_label_index]
    if predicted_label == prod_cat_class:
        container.markdown(':green[**Selected view (label) matches prediction.**]')
    else:
        container.markdown(':red[**Selected view (label) DOES NOT match prediction.**]')
        container.markdown(':red[Please make sure you have selected the correct view for this image]')

    left_column1.image(img)

    print('Before GradCAM object creation.')
    gc = gcam.GradCAM(model=model, classIdx=predicted_label_index)
    print('After GradCAM object creation.') 
    heatmap = gc.compute_heatmap(img_arr, verbose=True)
    print('After compute_heatmap.')
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
    print('After resize heatmap.')
    (heatmap, output) = gc.overlay_heatmap(heatmap, orig, alpha=0.45)
    print('After overlay heatmap.')
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    middle_column1.image(output)
    # middle_column1.markdown('**GradCAM image will come here**')

    df = pd.DataFrame({"probs": predictions[0]}).sort_values(by="probs", ascending=False).reset_index()
    print(df)
    right_column1.markdown(f'**Predicted View = {predicted_label}**\n\n'
                        'View propabilities: \n\n'
                        f'{view_labels.get(prod_cat)[df["index"][0]]} = {round(df["probs"][0]*100, 2)}%\n\n'
                        f'{view_labels.get(prod_cat)[df["index"][1]]} = {round(df["probs"][1]*100, 2)}%\n\n'
                        f'{view_labels.get(prod_cat)[df["index"][2]]} = {round(df["probs"][2]*100, 2)}%\n\n'
                        f'{view_labels.get(prod_cat)[df["index"][3]]} = {round(df["probs"][3]*100, 2)}%\n\n'
                        f'{view_labels.get(prod_cat)[df["index"][4]]} = {round(df["probs"][4]*100, 2)}%\n\n'

                        # f'{view_labels.get(prod_cat)[0]} = {round(predictions[0][0]*100, 2)}%\n\n'
                        # f'{view_labels.get(prod_cat)[1]} = {round(predictions[0][1]*100, 2)}%\n\n'
                        # f'{view_labels.get(prod_cat)[2]} = {round(predictions[0][2]*100, 2)}%\n\n'
                        # f'{view_labels.get(prod_cat)[3]} = {round(predictions[0][3]*100, 2)}%\n\n'
                        # f'{view_labels.get(prod_cat)[4]} = {round(predictions[0][4]*100, 2)}%\n\n'
                        )


             