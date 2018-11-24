import matplotlib.pyplot as plt
import imageio
import numpy as np
from collections import OrderedDict

kernels = OrderedDict({"Identity": [[0, 0, 0], [0., 1., 0.], [0., 0., 0.]],
                       "Laplacian": [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
                       "Left Sobel": [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
                       "Upper Sobel": [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
                       "Blur": [[1. / 16., 1. / 8., 1. / 16.], [1. / 8., 1. / 4., 1. / 8.],
                                [1. / 16., 1. / 8., 1. / 16.]]})


def apply3x3kernel(image, kernel):
    newimage = np.array(image)
    for m in range(1, image.shape[0] - 2):
        for n in range(1, image.shape[1] - 2):
            newelement = 0
            for i in range(0, 3):
                for j in range(0, 3):
                    newelement = newelement + image[m - 1 + i][n - 1 + j] * kernel[i][j]
            newimage[m][n] = newelement
    return newimage


arr = imageio.imread("data/dog.jpg")[:, :, 0].astype(np.float)

plt.figure(1)
j = 0
positions = [321, 322, 323, 324, 325, 326]
for key, value in kernels.items():
    plt.subplot(positions[j])
    out = apply3x3kernel(arr, value)
    plt.imshow(out, cmap=plt.get_cmap('binary_r'))
    j = j + 1
plt.show()
