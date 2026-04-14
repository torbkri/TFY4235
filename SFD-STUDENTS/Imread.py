#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%%
img = cv2.imread('exampleSignature.png', cv2.IMREAD_GRAYSCALE)

print(img.shape)
print(img.dtype)
print(img)
# %%
