import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from collections import defaultdict
from private.utilities import *
import os, time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


@timing
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    gray_image = grayscale(image)
    blur_gray = gaussian_blur(gray_image, 5)
    high_threshold, _ = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = 0.5*high_threshold
    edges = canny(blur_gray, low_threshold, high_threshold)
    height, width = image.shape[0], image.shape[1]
    vertices = np.array([[(0, height),
                          (480, 313),
                          (480, 313),
                          (width, height)]],
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 2  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    color_edges = np.dstack((edges, edges, edges))
    pancake_img = weighted_img(line_image, color_edges)
    return pancake_img

def trying_out_canny(image):
    plt.imshow(process_image(image))
    plt.show()


def test_process_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def to_image(image):
    print("Processing {image}".format(image=image))
    return (mpimg.imread(image)*255).astype('uint8') if 'png' in image else mpimg.imread(image)

if __name__ == '__main__':
    #trying_out_canny((mpimg.imread('../../images/exit-ramp.png')*255).astype('uint8'))
    directory = "../../images/"
    test_images = list(map(lambda file: directory + file, os.listdir(directory)))
    each(test_process_image, map(process_image, map(to_image, test_images[:2])))
