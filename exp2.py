import numpy as np
import cv2
import matplotlib.pyplot as plt
image =cv2.imread(r"E:\sem 7\CV LAB\dog.jpg",cv2.IMREAD_GRAYSCALE)
plt.imshow(image,cmap='gray')
  
cv2.imshow('image',image)

def ass_salt_pepper_noise(image,probability=0.3):
    noisy=image.copy()
    
    random=np.random.rand(*image.shape)
    noisy[random<probability/2]=0
    noisy[random>1-probability/2]=255
    return noisy

prob=0.2
noisy_image=ass_salt_pepper_noise(image,prob)
plt.imshow(noisy_image,cmap='gray')
plt.show()
