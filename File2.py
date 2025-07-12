import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  #type:ignore

st.title("🐟 Fish Image Classifier")

# Load pre-trained model
model = load_model(r'C:\pro3\fish_classifier_vgg16.keras')

# Define class names - make sure this matches your model's final Dense layer output
class_names = ['Fish1', 'Fish2', 'Fish3', 'Fish4', 'Fish5', 
               'Fish6', 'Fish7', 'Fish8', 'Fish9', 'Fish10', 'Fish11']

# Upload image
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    img = load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    prediction = model.predict(x)
    st.write("Raw prediction output:", prediction)

    # Handle binary classification (e.g., shape (1, 1)) vs multiclass (e.g., shape (1, 3))
    if prediction.shape[1] == 1:
        # Binary classification
        predicted_index = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] if predicted_index == 1 else 1 - prediction[0][0]
        # Ensure class_names has exactly 2 classes
        if len(class_names) == 2:
            predicted_class = class_names[predicted_index]
            st.write(f"### Predicted: {predicted_class} ({confidence * 100:.2f}%)")
        else:
            st.error("Error: Model predicts 2 classes but class_names list doesn't have exactly 2 entries.")
    else:
        # Multiclass classification
        if prediction.shape[1] == len(class_names):
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            predicted_class = class_names[predicted_index]
            st.write(f"### Predicted: {predicted_class} ({confidence * 100:.2f}%)")
        else:
            st.error(f"Mismatch: Model predicts {prediction.shape[1]} classes but class_names has {len(class_names)}.")