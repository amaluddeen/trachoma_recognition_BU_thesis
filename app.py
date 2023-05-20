import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pickle



def load_model_custom(model_name):
    if model_name == 'CNN':
        model = load_model('models/trachoma_cnn.h5')
    elif model_name == 'TRF':
        model = load_model('models/trachoma_trf_learning.h5')
    elif model_name == 'RF':
        model = pickle.load(open('models/trachoma_rf.sav', 'rb'))
    return model



def convert_to_rgb_and_resize(img):

    # Check the number of channels (bands) in the image
    if img.mode not in ("RGB", "RGBA"):
        # Convert grayscale or single channel images to RGB
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        # Convert RGBA images to RGB
        img = img.convert("RGB")

    # Resize the image to 128x128
    img = img.resize((128, 128), Image.ANTIALIAS)

    return img


def predict_image(model_name, image):
    # Convert the image to RGB and resize
    img = convert_to_rgb_and_resize(image)

    # Convert the image to a numpy array and expand dimensions
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image (pixel values to 0-1)
    img_array = img_array.astype('float32') / 255

    # Load model
    if model_name in ['CNN', 'TRF']:
        model = load_model_custom(model_name)
        prediction = model.predict(img_array)
        trachoma_prob = prediction[0]
    
    elif model_name == 'RF':
        model = load_model_custom(model_name)
        prediction = model.predict_proba(img_array.reshape(1, -1))
        trachoma_prob = 1-prediction[0]

    predict_text = f'Trachoma Probability: {round((trachoma_prob[0]*100), 2)} percent.'

    return predict_text




st.title('Trachoma Detection System')

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    selected_model = st.selectbox('Select Model', ['CNN', 'TRF', 'RF'])
    pred_button = st.button("Predict")

    if pred_button:
        with st.spinner("Classifying..."):
            label = predict_image(selected_model, image)
            st.info('%s' % label)

