import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# Read the input image and convert it to grayscale.
image_path = Path(__file__).with_name("dog.jpg")
image = cv2.imread(str(image_path))

if image is None:
	raise FileNotFoundError(f"Could not read image: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Segment the image using global, Otsu, and adaptive thresholding.
_, global_threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
otsu_value, otsu_threshold = cv2.threshold(
	gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
adaptive_threshold = cv2.adaptiveThreshold(
	gray,
	255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	cv2.THRESH_BINARY,
	11,
	2,
)

print(f"Otsu threshold value: {otsu_value:.0f}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Threshold")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Threshold")
plt.axis("off")

plt.tight_layout()
plt.show()
