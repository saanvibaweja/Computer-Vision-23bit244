import numpy as np
import cv2
import matplotlib.pyplot as plt
image =cv2.imread(r"E:\sem 7\CV LAB\dog.jpg",cv2.IMREAD_GRAYSCALE)
plt.imshow(image,cmap='gray')
  
cv2.imshow('image',image)

def add_salt_pepper_noise(image,probability=0.3):
    noisy=image.copy()
    
    random=np.random.rand(*image.shape)
    noisy[random<probability/2]=0
    noisy[random>1-probability/2]=255
    return noisy

# prob=0.2
# noisy_image=ass_salt_pepper_noise(image,prob)
# plt.imshow(noisy_image,cmap='gray')
# plt.show()
noisy_image = add_salt_pepper_noise(image, probability=0.2)

# Remove noise using Median Filter (Moving Window)
filtered_image = cv2.medianBlur(noisy_image, 3)

# Display images
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(noisy_image, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(filtered_image, cmap='gray')
plt.title("Median Filter Output")
plt.axis('off')

plt.tight_layout()
plt.show()

