import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the input image
image = cv2.imread(r"E:\sem 7\CV LAB\dog.jpg")

# Check if image is loaded
if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

# 2. Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert image to float for calculations
gray = gray.astype(np.float64)

# ---------------------------------------------------------
# 3. Define Sobel kernels manually
# ---------------------------------------------------------

sobel_x_kernel = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])

sobel_y_kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# ---------------------------------------------------------
# 4. Create empty images for Gx and Gy
# ---------------------------------------------------------

height, width = gray.shape

Gx = np.zeros((height, width))
Gy = np.zeros((height, width))

# ---------------------------------------------------------
# 5. Add padding around the image
# ---------------------------------------------------------

padded = np.pad(gray, ((1, 1), (1, 1)), mode='constant')

# ---------------------------------------------------------
# 6. Perform Sobel convolution manually
# ---------------------------------------------------------

for i in range(height):
    for j in range(width):

        # Take 3x3 neighborhood
        region = padded[i:i+3, j:j+3]

        # Calculate Gx
        Gx[i, j] = np.sum(region * sobel_x_kernel)

        # Calculate Gy
        Gy[i, j] = np.sum(region * sobel_y_kernel)

# ---------------------------------------------------------
# 7. Calculate gradient magnitude
# ---------------------------------------------------------

gradient_magnitude = np.sqrt(Gx**2 + Gy**2)

# ---------------------------------------------------------
# 8. Normalize values to 0-255
# ---------------------------------------------------------

gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
gradient_magnitude = gradient_magnitude.astype(np.uint8)

Gx_display = np.abs(Gx)
Gx_display = np.clip(Gx_display, 0, 255).astype(np.uint8)

Gy_display = np.abs(Gy)
Gy_display = np.clip(Gy_display, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------

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
plt.imshow(Gx_display, cmap="gray")
plt.title("Sobel X - Vertical Edges")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(Gy_display, cmap="gray")
plt.title("Sobel Y - Horizontal Edges")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gradient_magnitude, cmap="gray")
plt.title("Final Sobel Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()