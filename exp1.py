# ==========================================================
# IMAGE PROCESSING USING OPENCV
# ==========================================================

# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Load Image
# ----------------------------------------------------------
image_path = r"E:\sem 7\CV LAB\dog.jpg"
image = cv2.imread(image_path)

# Display original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------------------------------------
# Image Information
# ----------------------------------------------------------
print("Image Dimensions :", image.shape)
print("Data Type        :", image.dtype)

# ----------------------------------------------------------
# Crop Selected Region
# ----------------------------------------------------------
cropped_image = image[400:880, 90:700]

cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------------------------------------
# Resize Image
# ----------------------------------------------------------
resized_image = cv2.resize(image, (350, 550))

cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------------------------------------
# Modify a Pixel Value
# ----------------------------------------------------------
modified_image = image.copy()

# Change one pixel to black
modified_image[120, 100] = (0, 0, 0)

cv2.imshow("Modified Pixel", modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------------------------------------
# Convert to Grayscale
# ----------------------------------------------------------
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------------------------------------
# BGR vs RGB Comparison
# ----------------------------------------------------------
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("BGR Display")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rgb_image)
plt.title("RGB Display")
plt.axis("off")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Display Red Color Channel
# ----------------------------------------------------------
red_channel = image[:, :, 2]

plt.figure(figsize=(5,5))
plt.imshow(red_channel, cmap="gray")
plt.title("Red Channel")
plt.axis("off")
plt.show()

# ==========================================================
# Additional Image Operations
# ==========================================================

# Horizontal Flip
horizontal_flip = cv2.flip(image, 1)

cv2.imshow("Horizontal Flip", horizontal_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotate 90 Degrees Clockwise
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)

cv2.imshow("Gaussian Blur", gaussian_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
