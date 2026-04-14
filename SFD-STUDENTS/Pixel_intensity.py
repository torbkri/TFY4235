#%%
from Image_inversion import invert_image
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%%
def p_k(k:int , img) -> float:
    return np.sum(img == k) / img.size
# %%

img = cv2.imread('exampleSignature.png', cv2.IMREAD_GRAYSCALE)

print((img).size)
# %%

pmf = [p_k(k, img) for k in range(256)]

# %%
fig,axs = plt.subplots(1,2, figsize=(10,5))

axs[0].hist(img.ravel(), bins=16, range=(0,256))
axs[0].set_title('Histogram')
axs[0].set_xlabel('Pixel Intensity')
axs[0].set_ylabel('Frequency')

axs[1].hist(img.ravel(), bins=256, range=(0,256))
axs[1].set_title('Histogram')
axs[1].set_xlabel('Pixel Intensity')
axs[1].set_ylabel('Frequency')
# %%
#Small number of bins gives grouped pixel intensitites, and thus lower specificity. 
#Larger number takes longer to compute ans is more susceptible to noise.


# %%
