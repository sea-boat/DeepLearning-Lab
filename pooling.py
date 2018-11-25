import matplotlib.pyplot as plt
import imageio
import numpy as np


def apply2x2pooling(image, stride):
    newimage = np.zeros((int(image.shape[0] / 2), int(image.shape[1] / 2)), np.float32)
    for m in range(1, image.shape[0] - 2, 2):
        for n in range(1, image.shape[1] - 2, 2):
            newimage[int(m / 2), int(n / 2)] = np.max(image[m:m + 2, n:n + 2])
    return (newimage)


arr = imageio.imread("data/dog.jpg")[:, :, 0].astype(np.float)
plt.figure(1)
plt.subplot(121)
plt.imshow(arr, cmap=plt.get_cmap('binary_r'))
out = apply2x2pooling(arr, 1)
plt.subplot(122)
plt.imshow(out, cmap=plt.get_cmap('binary_r'))
plt.show()
