#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def rotate(image, theta) -> np.ndarray:
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), theta, 0.9)
    rotated = cv2.warpAffine(image, 
                             M = rotation_matrix, 
                             dsize=(image.shape[1], 
                                    image.shape[0]), 
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=255)
    return rotated

def signature_height(image):
    rows = np.where(image < 250)[0]
    if len(rows) == 0:
        return 0
    return rows.max() - rows.min()

#%%

img = cv2.imread("exampleSignature_diag.png", cv2.IMREAD_GRAYSCALE)
img = cv2.threshold(src = img, thresh= 185, maxval = 256, type = cv2.THRESH_BINARY)[1]

display_img(img)
print(signature_height(img))
# %%
rotated = rotate(img, -42)
display_img(rotated)
print(signature_height(rotated))
# %%
def fdm(image, increment) -> np.ndarray:
    minimized = image

    for angle in np.arange(-90,90,increment):
        attempt = rotate(image,angle)
        if signature_height(attempt) >= signature_height(minimized): pass
        else: minimized = attempt

    return minimized
# %%
aligned = fdm(img, 1)
aligned = cv2.threshold(src = aligned, thresh= 185, maxval = 256, type = cv2.THRESH_BINARY)[1]
# %%

display_img(aligned)
