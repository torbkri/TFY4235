#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('exampleSignature.png', cv2.IMREAD_GRAYSCALE)

#No the transform is not linera, as T(0) != 0

def invert_image(image) -> np.ndarray:
    invert_image: np.ndarray = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            invert_image[i][j] = 255 - image[i][j]
    return invert_image

fig,axs = plt.subplots(1,2, figsize=(10,5))
axs[0].hist(invert_image(img).ravel(), bins=16, range=(0,256))
axs[0].set_title('Histogram_inverted_pic')
axs[0].set_xlabel('Pixel Intensity')
axs[0].set_ylabel('Frequency')

axs[1].hist(invert_image(img).ravel(), bins=256, range=(0,256))
axs[1].set_title('Histogram_inverted_pic')
axs[1].set_xlabel('Pixel Intensity')
axs[1].set_ylabel('Frequency')



# %%
