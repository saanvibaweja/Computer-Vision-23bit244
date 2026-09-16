import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the input image
image = cv2.imread(r"E:\sem 7\CV LAB\dog.jpg")

# Check if image is loaded
if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

# --------------------------------------------------
# 2. Convert BGR image to RGB manually
# --------------------------------------------------

image_rgb = image[:, :, ::-1]

# --------------------------------------------------
# 3. Convert RGB image to Grayscale manually
# Formula:
# Gray = 0.299R + 0.587G + 0.114B
# --------------------------------------------------

B = image[:, :, 0].astype(np.float32)
G = image[:, :, 1].astype(np.float32)
R = image[:, :, 2].astype(np.float32)

gray = 0.299 * R + 0.587 * G + 0.114 * B
gray = np.clip(gray, 0, 255).astype(np.uint8)

# --------------------------------------------------
# 4. Generate Gaussian Noise
# Mean = 0
# Sigma = 25
# --------------------------------------------------

mean = 0
sigma_noise = 25

gaussian_noise = np.random.normal(
    mean,
    sigma_noise,
    gray.shape
).astype(np.float32)

# --------------------------------------------------
# 5. Add Gaussian Noise
# Noisy Image = Original + Noise
# --------------------------------------------------

noisy_image = gray.astype(np.float32) + gaussian_noise

# Keep values between 0 and 255
noisy_image = np.clip(
    noisy_image,
    0,
    255
).astype(np.uint8)

# --------------------------------------------------
# 6. Create Gaussian Kernel Manually
#
# Formula:
# G(x,y) = 1/(2*pi*sigma^2)
#          * exp(-(x^2+y^2)/(2*sigma^2))
# --------------------------------------------------

kernel_size = 5
sigma = 1

radius = kernel_size // 2

gaussian_kernel = np.zeros(
    (kernel_size, kernel_size),
    dtype=np.float32
)

for x in range(-radius, radius + 1):
    for y in range(-radius, radius + 1):

        gaussian_kernel[
            x + radius,
            y + radius
        ] = np.exp(
            -(x**2 + y**2) / (2 * sigma**2)
        )

# Normalize the kernel
gaussian_kernel = (
    gaussian_kernel /
    np.sum(gaussian_kernel)
)

print("Gaussian Kernel:")
print(gaussian_kernel)

print("Sum of Kernel:",
      np.sum(gaussian_kernel))

# --------------------------------------------------
# 7. Apply Gaussian Filter Manually
# Using convolution
# --------------------------------------------------

height, width = noisy_image.shape

filtered_image = np.zeros(
    (height, width),
    dtype=np.float32
)

# Padding required for 5x5 kernel
padded_image = np.pad(
    noisy_image,
    ((radius, radius), (radius, radius)),
    mode="reflect"
)

# Convolution
for i in range(height):
    for j in range(width):

        # Extract 5x5 neighborhood
        region = padded_image[
            i:i + kernel_size,
            j:j + kernel_size
        ]

        # Multiply neighborhood with Gaussian kernel
        multiplied = region * gaussian_kernel

        # Add all values
        filtered_image[i, j] = np.sum(multiplied)

# Convert to uint8
filtered_image = np.clip(
    filtered_image,
    0,
    255
).astype(np.uint8)

# --------------------------------------------------
# 8. Display Results
# --------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(noisy_image, cmap="gray")
plt.title("Image with Gaussian Noise")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(filtered_image, cmap="gray")
plt.title("Gaussian Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()