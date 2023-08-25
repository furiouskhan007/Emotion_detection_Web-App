import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from PIL import Image

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self, model_path):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = tf.keras.models.load_model(model_path)
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    def transform(self, frame):
        img = np.array(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_array = np.array(face)
            face_array = np.expand_dims(face_array, axis=-1)
            face_array = np.expand_dims(face_array, axis=0)
            prediction = np.argmax(self.model.predict(face_array), axis=1)
            emotion = self.emotion_labels[prediction[0]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, emotion, (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title('Face Expression Classifier')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_transformer = FaceDetectionTransformer('my_model.h5')

    st.image(image, caption='Uploaded Image', use_column_width=True)
    result_img = image_transformer.transform(image)
    st.image(result_img, caption='Result', use_column_width=True)
