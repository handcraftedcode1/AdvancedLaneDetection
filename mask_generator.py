from scipy.misc import imread
import cv2
import numpy as np
from utils.plot_utils import *


def abs_sobel(img, orient='x', sobel_kernel=3):
    """
    Apply the sobel operation on a gray scale image

    :param img: Input image
    :param orient: 'x' or 'y'
    :param sobel_kernel: A non-even integer (2, 4, 6, 8, ...)
    :return:
    """

    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)

    sobel = cv2.Sobel(img, -1, *axis, ksize=sobel_kernel)
    abs_s = np.absolute(sobel)

    return abs_s


def gradient_magnitude(sobel_x, sobel_y):
    """
    Calculate the magnitude of the gradient.

    :param sobel_x: List of vector magnitudes in x-direction
    :param sobel_y: List of vector magnitudes in y-direction
    :return: Magitude of overall
    """
    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return abs_grad_mag.astype(np.uint16)


def gradient_direction(sobel_x, sobel_y):
    """
    Calculate the direction of the gradient.
    NaN values cause by zero division will be replaced by a maximum value pi/2

    :param sobel_x: List of vector magnitudes in x-direction
    :param sobel_y: List of vector magnitudes in y-direction
    :return:
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)


def gaussian_blur(img, kernel_size):
    """
    Apply a Gaussian blur

    :param img: Input image to blur
    :param kernel_size: Size of kernel to convolve image with
    :return: A gaussian blurred image
    """

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def extract_yellow(img):
    """
    Create an image mask by selecting a range of yellow pixels

    :param img: RGB image
    :return: Grayscale image masking yellow colors
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))

    return mask


def extract_dark(img):
    """
    Generate an image mask selecting "dark" pixels

    :param img: RGB image
    :return: Grayscale image masking dark colors
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0.), (255, 153, 128))

    return mask


def extract_highlights(img, p=99.9):
    """
    Generate an image mask selecting highlights

    :param p: A percentile for highlight selection. default=99.9
    :param img: RGB image
    :return: Grayscale image masking colors falling within a given percentile threshold
    """

    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask


def binary_noise_reduction(img, thresh):
    """
    Reduce noise of a binary image by applying a filter which counts neighbors with a value
    and only keeping those which are above the threshold

    :param img: Binary image mask
    :param thresh: Minimum number of neighbors with a non-zero value
    :return:
    """

    k = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]]
    )

    nb_neighbors = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbors < thresh] = 0

    return img


def generate_lane_mask(img, v_cutoff=0):
    """
    Generate a binary mask selecting the lane lines of an street scene image

    :param img: RGB image
    :param v_cutoff: vertical cutoff to limit the search area
    :return: binary mask
    """

    window = img[v_cutoff:, :, :]

    yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
    yuv = 255 - yuv
    hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
    chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
    gray = np.mean(chs, 2)

    s_x = abs_sobel(gray, orient='x', sobel_kernel=3)
    s_y = abs_sobel(gray, orient='y', sobel_kernel=3)

    grad_dir = gradient_direction(s_x, s_y)
    grad_mag = gradient_magnitude(s_x, s_y)

    ylw = extract_yellow(window)
    highlights = extract_highlights(window[:, :, 0])

    mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &
                        (s_y >= 25) & (s_y <= 255)) |
                       ((grad_mag >= 30) & (grad_mag <= 512) &
                        (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                       (ylw == 255) |
                       (highlights == 255)] = 1

    mask = binary_noise_reduction(mask, 4)

    return mask


def outlier_removal(x, y, q=5):
    """
    Remove horizontal outliers based on a given percentile

    :param x: x coordinates of pixels
    :param y: y coordinates of pixels
    :param q: percentile
    :return: cleaned coordinates (x, y)
    """

    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)

    return x[selection], y[selection]


if __name__ == '__main__':

    images_left = []
    images_right = []

    test_images = ["./test_images/straight_lines2.jpg", "./test_images/test5.jpg"]

    for image in test_images:
        image = imread(image)

        mask = generate_lane_mask(image, 400)
        images_left.append(image)
        images_right.append(mask)

    plot_side_by_side_images(images_left, images_right, 'Original', 'Combined Mask',
                             './output_images/combined_mask.png')

    print('Done')
