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

    # Pepper noise
    noisy_image[random_values < probability / 2] = 0

    # Salt noise
    noisy_image[random_values > 1 - probability / 2] = 255

    return noisy_image


def mean_filter(image, window_size=3, stride=1):
    """Apply a manual mean filter with given kernel size and stride."""

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number.")

    if stride < 1:
        raise ValueError("stride must be a positive integer.")

    padding = window_size // 2

    padded_image = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_REFLECT
    )

    height, width = image.shape

    # Output size considering stride
    output_height = (height - 1) // stride + 1
    output_width = (width - 1) // stride + 1

    filtered_image = np.zeros(
        (output_height, output_width),
        dtype=np.uint8
    )

    output_row = 0

    for row in range(0, height, stride):

        output_column = 0

        for column in range(0, width, stride):

            window = padded_image[
                row:row + window_size,
                column:column + window_size
            ]

            mean_value = np.mean(window)

            filtered_image[output_row, output_column] = mean_value

            output_column += 1

        output_row += 1

    return filtered_image


def median_filter(image, window_size=3, stride=1):
    """Apply a manual median filter with given kernel size and stride."""

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number.")

    if stride < 1:
        raise ValueError("stride must be a positive integer.")

    padding = window_size // 2

    padded_image = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_REFLECT
    )

    height, width = image.shape

    # Output size considering stride
    output_height = (height - 1) // stride + 1
    output_width = (width - 1) // stride + 1

    filtered_image = np.zeros(
        (output_height, output_width),
        dtype=np.uint8
    )

    output_row = 0

    for row in range(0, height, stride):

        output_column = 0

        for column in range(0, width, stride):

            window = padded_image[
                row:row + window_size,
                column:column + window_size
            ]

            median_value = np.median(window)

            filtered_image[output_row, output_column] = median_value

            output_column += 1

        output_row += 1

    return filtered_image


def main(image_path, probability):

    # Read image as grayscale
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Add salt-and-pepper noise
    noisy_image = add_salt_pepper_noise(
        image,
        probability
    )

    # ------------------------------------------------
    # 3x3 kernel, stride 1
    # ------------------------------------------------

    mean_3_s1 = mean_filter(
        noisy_image,
        window_size=3,
        stride=1
    )

    median_3_s1 = median_filter(
        noisy_image,
        window_size=3,
        stride=1
    )

    # ------------------------------------------------
    # 5x5 kernel, stride 1
    # ------------------------------------------------

    mean_5_s1 = mean_filter(
        noisy_image,
        window_size=5,
        stride=1
    )

    median_5_s1 = median_filter(
        noisy_image,
        window_size=5,
        stride=1
    )

    # ------------------------------------------------
    # 3x3 kernel, stride 2
    # ------------------------------------------------

    mean_3_s2 = mean_filter(
        noisy_image,
        window_size=3,
        stride=2
    )

    median_3_s2 = median_filter(
        noisy_image,
        window_size=3,
        stride=2
    )

    # ------------------------------------------------
    # Display results
    # ------------------------------------------------

    plt.figure(figsize=(15, 10))

    images = [
        ("Original Image", image),
        ("Noisy Image", noisy_image),

        ("Mean 3x3, Stride 1", mean_3_s1),
        ("Median 3x3, Stride 1", median_3_s1),

        ("Mean 5x5, Stride 1", mean_5_s1),
        ("Median 5x5, Stride 1", median_5_s1),

        ("Mean 3x3, Stride 2", mean_3_s2),
        ("Median 3x3, Stride 2", median_3_s2),
    ]

    for position, (title, display_image) in enumerate(
        images,
        start=1
    ):

        plt.subplot(2, 4, position)

        plt.imshow(
            display_image,
            cmap="gray"
        )

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Mean and median filtering with different kernel sizes and strides."
    )

    parser.add_argument(
        "image",
        nargs="?",
        default="dog.jpg",
        help="Path to input image"
    )

    parser.add_argument(
        "--probability",
        type=float,
        default=0.04,
        help="Salt-and-pepper noise probability"
    )

    arguments = parser.parse_args()

    main(
        arguments.image,
        arguments.probability
    )