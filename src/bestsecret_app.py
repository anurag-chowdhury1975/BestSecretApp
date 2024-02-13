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
from tensorflow.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from struct import unpack
from tqdm import tqdm

# @st.cache_data    
def load_models():
    models = {}
    model_bag = load_model('models/bag_resnet50_model_ft_all_93%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_clothes = load_model('models/clothes_resnet50_func_model_97%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_schuhe = load_model('models/schuhe_resnet50_model_ft_all_94%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_waesche = load_model('models/waesch_funcResnet_model_94%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    models['bag'] = model_bag
    models['clothes'] = model_clothes
    models['schuhe'] = model_schuhe
    models['waesche'] = model_waesche
    return models

def image_processing_function(im_path, input_img_dims, pre_process_function=None):

    orig = image.load_img(im_path)
    orig_arr = image.img_to_array(orig).astype("uint8")
    img = image.load_img(im_path, target_size=input_img_dims)

    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)

    return img, image_arr, orig_arr

def eval_model_on_test(model, test_ds):

    test_labels = []
    predictions = []

    for imgs, labels in tqdm(test_ds.take(1000),
                             desc='Predicting on Test Data'):
        batch_preds = model.predict(imgs)
        predictions.extend(batch_preds)
        test_labels.extend(labels)
    if len(predictions[0]) > 1:
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = np.array(predictions)

    test_labels = np.array(test_labels)

    return test_labels, predictions

models = load_models()
# explainer = GradCAM()

model = ""

prod_labels = {
    "bag" : "Bags",
    "clothes" : "Clothes",
    "schuhe" : "Schuhe",
    "waesche" : "Waesche"
}

view_labels = {
    "bag": ["Front","Side","Inside","Back","Look"],
    "clothes": ["Model Front","Zoomed","Model Back","Ghost","Look"],
    "schuhe": ["Overll to Right","Back","Top or Sole","Side to Left","Zoom"],
    "waesche": ["Model Front","Zoomed","Model Back","Ghost","Look"]
}

def format_func(option):
    return prod_labels[option]

st.set_page_config(page_title="Image View Classifier", layout="wide")
st.header("Image View Classifier")
prod_cat = st.selectbox(label="Select Product Category", options=list(prod_labels.keys()), format_func=format_func)
model = models.get(prod_cat)

tab1, tab2 = st.tabs(["Single Image Prediction", "Prediction Performance on Test Dataset"])

with tab1:
    prod_cat_class = st.selectbox(label="Select Product View", options=view_labels.get(prod_cat))
    uploaded_image = tab1.file_uploader("Please choose an image...", type=["jpg", "jpeg", "png"])

    container = tab1.container(border=True)
    left_column1, middle_column1, right_column1 = tab1.columns([1,1,1])

    if uploaded_image is not None:
        input_img_dims = (427, 350)
        img, img_arr, orig = image_processing_function(uploaded_image, input_img_dims)
        predictions = model.predict(img_arr)
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        predicted_label = view_labels.get(prod_cat)[predicted_label_index]
        if predicted_label == prod_cat_class:
            container.markdown(':green[**Selected view (label) matches prediction.**]')
        else:
            container.markdown(':red[**Selected view (label) DOES NOT match prediction.**]')
            container.markdown(':red[Please make sure you have selected the correct view for this image]')

        left_column1.image(img)

        gc = gcam.GradCAM(model=model, classIdx=predicted_label_index)
        heatmap = gc.compute_heatmap(img_arr, verbose=True)
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
        (heatmap, output) = gc.overlay_heatmap(heatmap, orig, alpha=0.45)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        middle_column1.image(output)
        # middle_column1.markdown('**GradCAM image will come here**')

        df = pd.DataFrame({"probs": predictions[0]}).sort_values(by="probs", ascending=False).reset_index()
        right_column1.markdown(f'**Predicted View = {predicted_label}**\n\n'
                            'View propabilities: \n\n'
                            f'{view_labels.get(prod_cat)[df["index"][0]]} = {round(df["probs"][0]*100, 2)}%\n\n'
                            f'{view_labels.get(prod_cat)[df["index"][1]]} = {round(df["probs"][1]*100, 2)}%\n\n'
                            f'{view_labels.get(prod_cat)[df["index"][2]]} = {round(df["probs"][2]*100, 2)}%\n\n'
                            f'{view_labels.get(prod_cat)[df["index"][3]]} = {round(df["probs"][3]*100, 2)}%\n\n'
                            f'{view_labels.get(prod_cat)[df["index"][4]]} = {round(df["probs"][4]*100, 2)}%\n\n'
                            )

with tab2:
    test_ds = Dataset.load(f'data/{prod_cat}/test_dataset')
    if tab2.button("Run Prediction on Test Dataset"):
        y_true, y_pred = eval_model_on_test(model, test_ds)
        score = (accuracy_score(y_true, y_pred)*100)
        tab2.markdown("\n\nAccuracy of base model on test data: **%.2f%%**" % score)
        tab2.markdown("\n\n**Classification Report:**")
        report = classification_report(y_true, y_pred, output_dict=True)
        tab2.write(pd.DataFrame(report).transpose()) #, target_names=CLASSES
        tab2.markdown("\n\n**Confusion Matrix:**")
        tab2.write(pd.DataFrame(confusion_matrix(y_true, y_pred)))

             