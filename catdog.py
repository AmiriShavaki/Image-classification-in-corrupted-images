import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

def predict(image_file):
    loaded_model = load_model('acc_8102.keras')

    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)	

    prediction = loaded_model.predict(img)
    label = 'dog' if prediction > 0.5 else 'cat'
    return label, prediction
    
