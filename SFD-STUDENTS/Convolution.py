#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_img(image):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Image')
    ax[1].hist(image.ravel(), bins=256, range=(0, 256))
    ax[1].set_title('Histogram')
    ax[1].set_xlabel('Pixel Intensity')
    ax[1].set_ylabel('Frequency (log scale)')
    ax[1].set_yscale('log')
    fig.tight_layout()
    plt.show()
    plt.close()

A: np.ndarray =1/9 * np.array([[1,1,1],
                               [1,1,1],
                               [1,1,1]])

B: np.ndarray = np.array([[0,-1,0],
                          [-1,5,-1],
                          [0,-1,0]])

C: np.ndarray = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])

#%%
def convolution(image: np.ndarray, kernel: np.ndarray)->np.ndarray:
    convoluted: np.ndarray = np.zeros(image.shape, dtype=np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    try:
                        convoluted[x][y] += image[x+i-1][y+j-1] * kernel[i][j]
                    except:
                        pass
    return convoluted

#%%
#Testing the convolution kernel on images:
img = cv2.imread('exampleSignature.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
# %%
conv = convolution(img,A)

# %%

fig,ax = plt.subplots(1,2, figsize=(10,5))



print(conv)
ax[0].imshow(img, cmap = 'Greys')# plt.imshow(conv)
ax[1].imshow(conv, cmap = 'Greys')# plt.imshow(conv)


# %%
