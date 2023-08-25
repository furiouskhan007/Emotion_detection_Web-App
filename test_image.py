import cv2
import numpy as np
import tensorflow as tf

class ImageDetection(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = tf.keras.models.load_model('my_model.h5')
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    def detect_emotion(self, image_path, output_path):
        self.img = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)

        for (x, y, w, h) in self.faces:
            self.face = self.gray[y:y+h, x:x+w]
            self.face = cv2.resize(self.face, (48, 48))
            self.face_array = np.array(self.face)
            self.face_array = np.expand_dims(self.face_array, axis=-1)
            self.face_array = np.expand_dims(self.face_array, axis=0)
            self.prediction = np.argmax(self.model.predict(self.face_array), axis=1)
            self.emotion = self.emotion_labels[self.prediction[0]]
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, self.emotion, (x, y), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Face Expression Classifier', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the result image
        cv2.imwrite(output_path, self.img)
        print(f"Result image saved at: {output_path}")

if __name__ == "__main__":
    image_detection = ImageDetection()
    image_path = '6.jpg'  # Replace with the path to your image
    output_path = 'r6.jpg'  # Replace with the desired output path
    image_detection.detect_emotion(image_path, output_path)
