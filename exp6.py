import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. Read the input image
# ---------------------------------------------------------

image_path = Path(__file__).with_name("dog.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")


# ---------------------------------------------------------
# 2. Convert image to grayscale
# ---------------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to NumPy array with integer values
gray = gray.astype(np.uint8)


# =========================================================
# 3. GLOBAL THRESHOLDING
# =========================================================

T = 127

# Formula:
#
# g(x,y) = 255, if f(x,y) > T
#          0,   otherwise

global_threshold = np.zeros_like(gray)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):

        if gray[i, j] > T:
            global_threshold[i, j] = 255
        else:
            global_threshold[i, j] = 0


# =========================================================
# 4. OTSU THRESHOLDING
# =========================================================

# Calculate histogram
histogram = np.zeros(256, dtype=np.int64)

for pixel in gray.flatten():
    histogram[pixel] += 1


total_pixels = gray.size

# Total intensity sum
total_sum = 0

for i in range(256):
    total_sum += i * histogram[i]


# Variables used for Otsu calculation
weight_background = 0
sum_background = 0

maximum_variance = 0
otsu_value = 0


# Try every possible threshold from 0 to 255
for threshold in range(256):

    # Weight of background class
    weight_background += histogram[threshold]

    if weight_background == 0:
        continue

    # Weight of foreground class
    weight_foreground = total_pixels - weight_background

    if weight_foreground == 0:
        break

    # Sum of intensity values in background
    sum_background += threshold * histogram[threshold]

    # Mean of background
    mean_background = sum_background / weight_background

    # Mean of foreground
    sum_foreground = total_sum - sum_background
    mean_foreground = sum_foreground / weight_foreground

    # Otsu's between-class variance
    between_class_variance = (
        weight_background / total_pixels
        * weight_foreground / total_pixels
        * (mean_background - mean_foreground) ** 2
    )

    # Find maximum variance
    if between_class_variance > maximum_variance:
        maximum_variance = between_class_variance
        otsu_value = threshold


# Apply the optimum Otsu threshold
otsu_threshold = np.zeros_like(gray)

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):

        if gray[i, j] > otsu_value:
            otsu_threshold[i, j] = 255
        else:
            otsu_threshold[i, j] = 0


# =========================================================
# 5. ADAPTIVE GAUSSIAN THRESHOLDING
# =========================================================

block_size = 11
C = 2

# Radius around the center pixel
radius = block_size // 2

# Pad image so boundary pixels can also be processed
padded = np.pad(
    gray,
    ((radius, radius), (radius, radius)),
    mode="reflect"
)

adaptive_threshold = np.zeros_like(gray)


# ---------------------------------------------------------
# Create Gaussian kernel
# ---------------------------------------------------------

sigma = 3

gaussian_kernel = np.zeros((block_size, block_size), dtype=np.float64)

for x in range(-radius, radius + 1):
    for y in range(-radius, radius + 1):

        # Gaussian formula:
        #
        # G(x,y) = 1/(2πσ²) *
        #          exp(-(x²+y²)/(2σ²))

        gaussian_kernel[
            x + radius,
            y + radius
        ] = np.exp(
            -(x**2 + y**2) / (2 * sigma**2)
        )


# Normalize Gaussian kernel
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)


# ---------------------------------------------------------
# Calculate local threshold for every pixel
# ---------------------------------------------------------

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):

        # Extract local 11 × 11 neighborhood
        region = padded[
            i:i + block_size,
            j:j + block_size
        ]

        # Gaussian weighted mean
        local_mean = np.sum(region * gaussian_kernel)

        # Adaptive threshold formula:
        #
        # T(x,y) = Gaussian weighted mean - C

        local_threshold = local_mean - C

        # Binary thresholding
        if gray[i, j] > local_threshold:
            adaptive_threshold[i, j] = 255
        else:
            adaptive_threshold[i, j] = 0


# =========================================================
# 6. PRINT OTSU RESULT
# =========================================================

print("Otsu Optimum Threshold:", otsu_value)
print("Maximum Between-Class Variance:", maximum_variance)


# =========================================================
# 7. DISPLAY RESULTS
# =========================================================

plt.figure(figsize=(12, 8))


plt.subplot(2, 3, 1)

plt.imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)

plt.title("Original Image")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    gray,
    cmap="gray"
)

plt.title("Grayscale Image")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    global_threshold,
    cmap="gray"
)

plt.title("Global Threshold")
plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    otsu_threshold,
    cmap="gray"
)

plt.title("Otsu Threshold")
plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    adaptive_threshold,
    cmap="gray"
)

plt.title("Adaptive Gaussian Threshold")
plt.axis("off")


plt.tight_layout()
plt.show()