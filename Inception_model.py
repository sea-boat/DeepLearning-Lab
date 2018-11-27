from keras.applications.inception_v3 import InceptionV3, decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = InceptionV3()

im = cv2.resize(cv2.imread('data/dog.jpg'), (299, 299))
im = im / 255.
im = im - 0.5
im = im * 2
plt.figure()
plt.imshow(im)
plt.show()

im = np.expand_dims(im, axis=0)
out = model.predict(im)
print('Predicted:', decode_predictions(out, top=3)[0])
print(np.argmax(out))
