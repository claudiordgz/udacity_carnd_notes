import cv2
import numpy as np
from collections import defaultdict

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def each(fn, items):
    # map with no return
    for item in items:
        fn(item)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def _draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    def slope(x1, y1, x2, y2):
        return (y1 - y2) / (x1 - x2)

    def angle_between(m):
        r = np.arctan(m)
        return np.rad2deg((r) % (2 * np.pi))

    def truncate(num):
        num = int(num)
        leftover = num % 20
        return num - leftover if leftover < 16 else num - leftover + 20

    def perpendicular(a):
        b = np.empty_like(a)
        b[0], b[1] = -a[1], a[0]
        return b

    def lines_intersection_point(a1, a2, b1, b2):
        da, db, dp = a2 - a1, b2 - b1, a1 - b1
        dap = perpendicular(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        return (num / denom.astype(float)) * db + b1


    def get_interpolated_line(k, line_properties):
        n = len(line_properties[k])
        cords = np.sum([(i[2],i[3],i[4],i[5]) for i in line_properties[k]], axis=0)
        avg_coords = cords / n
        height, width, ypos = img.shape[0], img.shape[1], 330
        max_threshold_0, max_threshold_1 = np.array([0, height]), np.array([width, height])
        min_threshold_0, min_threshold_1 = np.array([0, ypos]), np.array([width, ypos])
        avg_0, avg_1 = np.array([avg_coords[0], avg_coords[1]]), np.array([avg_coords[2], avg_coords[3]])
        p_min = (lines_intersection_point(avg_0, avg_1, max_threshold_0, max_threshold_1))
        p_max = (lines_intersection_point(avg_0, avg_1, min_threshold_0, min_threshold_1))
        final_line = [p_min[0], p_min[1], p_max[0], p_max[1]]
        try:
            return list(map(int, final_line))
        except:
            return None

    def get_each_line_properties():
        line_properties = defaultdict(list)
        def line_attr(x1, y1, x2, y2):
            m = slope(x1, y1, x2, y2)
            angle = angle_between(m)
            k = truncate(angle)
            line_properties[k].append([angle, m, x1, y1, x2, y2])

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_attr(x1, y1, x2, y2)

        # more than two lines found
        if len(line_properties) > 2:
            bins = sorted([(k, len(line_properties[k])) for k in line_properties], key=lambda x: x[1])
            final_bins = [bins.pop()[0]]
            f_k = final_bins[-1]
            for bin in reversed(bins):
                k = bin[0]
                if f_k - 50 <= k <= f_k + 50:
                    pass
                else:
                    final_bins.append(k)
                    break
            final_properties = dict()
            while final_bins:
                bin = final_bins.pop()
                final_properties[bin] = line_properties[bin]
            return final_properties
        else:
            return line_properties

    lines_dict = get_each_line_properties()
    new_lines = [x for x in [get_interpolated_line(k, lines_dict) for k in lines_dict] if x is not None]
    for line in new_lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    _draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)