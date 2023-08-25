import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = tf.keras.models.load_model('my_model.h5')
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
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

webrtc_ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=FaceDetectionTransformer,
    rtc_configuration=rtc_config
)

if webrtc_ctx.video_transformer:
    st.write('Face detection is running...')
