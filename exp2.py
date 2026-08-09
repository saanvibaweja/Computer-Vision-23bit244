import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r"E:\sem 7\CV LAB\dog.jpg"
image = cv2.imread(image_path)
def add_salt_pepper_noise(image, probability=0.04):
    """Add salt (white) and pepper (black) noise to a grayscale image."""
    noisy_image = image.copy()
    random_values = np.random.rand(*image.shape)

    noisy_image[random_values < probability / 2] = 0
    noisy_image[random_values > 1 - probability / 2] = 255
    return noisy_image


def median_filter(image, window_size=3):
    """Remove noise using a manual moving-window median filter."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number.")

    padding = window_size // 2
    padded_image = cv2.copyMakeBorder(
        image, padding, padding, padding, padding, cv2.BORDER_REFLECT
    )
    filtered_image = np.zeros_like(image)

    height, width = image.shape
    for row in range(height):
        for column in range(width):
            window = padded_image[row : row + window_size, column : column + window_size]
            filtered_image[row, column] = np.median(window)

    return filtered_image


def main(image_path, probability):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    noisy_image = add_salt_pepper_noise(image, probability)
    denoised_image = median_filter(noisy_image, window_size=3)

    plt.figure(figsize=(15, 4))
    for position, (title, display_image) in enumerate(
        [
            ("Original Image", image),
            (f"Noisy Image (p={probability})", noisy_image),
            ("Denoised Image", denoised_image),
        ],
        start=1,
    ):
        plt.subplot(1, 3, position)
        plt.imshow(display_image, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salt-and-pepper noise removal.")
    parser.add_argument("image", nargs="?", default="dog.jpg", help="Path to input image")
    parser.add_argument("--probability", type=float, default=0.04, help="Noise probability")
    arguments = parser.parse_args()
    main(arguments.image, arguments.probability)
