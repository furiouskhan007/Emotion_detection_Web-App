import cv2
import numpy as np
import tensorflow as tf

class LiveDetection(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = tf.keras.models.load_model('my_model.h5')
        self.cap = cv2.VideoCapture(0)
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    def run(self):
        while True:
            _, self.img = self.cap.read()
            self.img = cv2.flip(self.img, 1)
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)

            for (x, y, w, h) in self.faces:
                self.face = self.gray[y:y+h, x:x+w]  # Use grayscale image for face detection
                self.face = cv2.resize(self.face, (48, 48))
                self.face_array = np.array(self.face)
                self.face_array = np.expand_dims(self.face_array, axis=-1)  # Add channel dimension
                self.face_array = np.expand_dims(self.face_array, axis=0)
                self.prediction = np.argmax(self.model.predict(self.face_array), axis=1)
                self.emotion = self.emotion_labels[self.prediction[0]]
                self.font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.img, self.emotion, (x, y), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Face Expression Classifier', self.img)
            self.k = cv2.waitKey(1) & 0xff
            if self.k == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    live_detection = LiveDetection()
    live_detection.run()
