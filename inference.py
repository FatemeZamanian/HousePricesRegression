import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('CNN_HousePrices.h5')
images = []
outImage = np.zeros((64, 64, 3), dtype="uint8")

for image in os.listdir('test'):
    img = cv2.imread('test/' + image)
    img = cv2.resize(img, (32, 32))
    images.append(img)

outImage[0:32, 0:32] = images[0]
outImage[0:32, 32:64] = images[1]
outImage[32:64, 32:64] = images[2]
outImage[32:64, 0:32] = images[3]

outImage = outImage/255
outImage = outImage.reshape(1, 64, 64, 3)
pred = model.predict([outImage])
print('House Price : ', pred)