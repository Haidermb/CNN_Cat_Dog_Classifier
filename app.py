import streamlit as st
from keras.preprocessing import image
from keras.models import load_model
import cv2 ,numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

#test_img = cv2.imread('/content/cat.jpg')

# Load the trained model
# with open('model2.pickle', 'rb') as file:
#     model = pickle.load(file)

model = load_model('model2.h5')

# Define labels for binary classification
class_labels = ['Cat', 'Dog']

# Streamlit app
st.title('Dog and Cat Classification Project')
st.write('Upload an image and I will predict whether it\'s a dog or a cat!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])




if uploaded_file is not None:

    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    test_img = cv2.imdecode(file_bytes, 1)
    
    # Preprocess the image
    test_img = cv2.resize(test_img, (256,256))
    test_input = test_img.reshape((1, 256,256, 3)) / 255.0
    
    # Make prediction using the model
    prediction = model.predict(test_input)
    predicted_class = class_labels[int(prediction[0][0] > 0.5)]
    
    # Display the uploaded image and prediction
    st.image(test_img, caption=f'Uploaded Image: {predicted_class}', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')
    
    # Display the image using Matplotlib
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {predicted_class}')
    plt.axis('off')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



