import numpy as np
from PIL import Image
import scipy.ndimage

############### ---------- Basic Image Processing ------ ##############

# TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image


def imread(filename):
    img = Image.open(filename)
    img_array = np.array(img).astype(np.float64)
    return img_array / 255.


# TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):
    m = len(img)  # image rows
    n = len(img[0])  # image columns

    filt = np.flip(filt, axis=0)
    l = len(filt)  # filter rows
    padH = (l-1)//2

    if len(np.shape(filt)) > 1:
        filt = np.flip(filt, axis=1)
        k = len(filt[0])  # filter cols
        padW = (k-1)//2
    else:
        k = 1
        padW = 0

    if len(np.shape(img)) == 2:
        padded = np.pad(img, ((padH, padH), (padW, padW)), mode='constant')
        init = np.zeros((m, n))
    else:
        padded = np.pad(
            img, ((padH, padH), (padW, padW), (0, 0)), mode='constant')
        init = np.zeros((m, n, 3))

    def filter(i, j):
        ans = 0.
        for x in range(-l // 2 + 1, l // 2 + 1):
            for y in range(-k // 2 + 1, k // 2 + 1):
                fx, fy = x + l // 2, y + k // 2
                if len(np.shape(filt)) > 1:
                    ans += filt[fx][fy] * padded[i + x + padH][j + y + padW]
                else:
                    ans += filt[fx] * padded[i + x + padH][j + y + padW]
        return ans

    for i in range(m):
        for j in range(n):
            init[i][j] = filter(i, j)

    return init


# TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    def gaussian(x, y):
        exponent = - (x ** 2 + y ** 2) / (2 * sigma ** 2)
        return np.exp(exponent)

    def coord(i, j):
        x = abs(k // 2 - i)
        y = abs(k // 2 - j)
        return x, y

    filter = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            x, y = coord(i, j)
            filter[i][j] = gaussian(x, y)

    filter /= np.sum(filter)

    return filter


# TODO 4: Compute the image gradient.
# First convert the image to grayscale by using the formula:
# Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
# Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise.
# Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
# convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
# Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    m = len(img)
    n = len(img[0])

    grayscale = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            r = img[i][j][0]
            g = img[i][j][1]
            b = img[i][j][2]

            grayscale[i][j] = 0.2125 * r + 0.7154 * g + 0.0721 * b

    filt = gaussian_filter(5, 1)
    grayscale = convolve(grayscale, filt)
    x_gradient = convolve(grayscale, [[0.5, 0, -0.5]])
    y_gradient = convolve(grayscale, [[0.5], [0], [-0.5]])

    grad_mag = np.sqrt(x_gradient[:, :] ** 2 + y_gradient[:, :] ** 2)
    grad_ori = np.arctan2(y_gradient[:, :], x_gradient[:, :])

    return grad_mag, grad_ori

# ----------------Line detection----------------

# TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
# x cos(theta) + y sin(theta) + c = 0
# The input x and y are numpy arrays of the same shape, representing the x and y coordinates of each pixel
# Return a boolean array that indicates True for pixels whose distance is less than the threshold


def check_distance_from_line(x, y, theta, c, thresh):
    # if not isinstance(x, np.ndarray):
    #     x = np.array(x)
    # if not isinstance(y, np.ndarray):
    #     y = np.array(y)
    d = abs(x * np.cos(theta) + y * np.sin(theta) + c)
    return d < thresh


# TODO 6: Write a function to draw a set of lines on the image.
# The `img` input is a numpy array of shape (m x n x 3).
# The `lines` input is a list of (theta, c) pairs.
# Mark the pixels that are less than `thresh` units away from the line with red color,
# and return a copy of the `img` with lines.
def draw_lines(img, lines, thresh):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    m, n, _ = np.shape(img)
    res = img.copy()

    xs, ys = np.meshgrid(np.arange(n), np.arange(m))
    # , indexing='xy'

    for theta, c in lines:
        red = check_distance_from_line(
            xs.ravel(), ys.ravel(), theta, c, thresh)
        red = red.reshape((m, n))

        for x in range(m):
            for y in range(n):
                if red[x, y]:
                    res[x, y] = [1, 0, 0]
    return res

    # img_copy = img.copy()
    # x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    # for (theta, c) in lines:
    #     i = np.where(check_distance_from_line(x, y, theta, c, thresh))[0]
    #     img_copy[x[i], y[i]] = [1, 0, 0]
    # return img_copy


# TODO 7: Do Hough voting. You get as input the gradient magnitude (m x n) and the gradient orientation (m x n),
# as well as a set of possible theta values and a set of possible c values.
# If there are T entries in thetas and C entries in cs, the output should be a T x C array.
# Each pixel in the image should vote for (theta, c) if:
# (a) Its gradient magnitude is greater than thresh1, **and**
# (b) Its distance from the (theta, c) line is less than thresh2, **and**
# (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    xpos, ypos = np.where(gradmag > thresh1)

    hough_vote = np.zeros((len(thetas), len(cs)))

    for i, theta in enumerate(thetas):
        for j, c in enumerate(cs):
            close_enough = check_distance_from_line(
                ypos, xpos, theta, c, thresh2)

            # for k, valid in enumerate(close_enough):
            #     if valid:
            #         x, y = xpos[k], ypos[k]
            #         if (abs(theta - gradori[x, y]) < (thresh3)):
            #             hough_vote[i, j] += 1
            o = gradori[xpos, ypos] - theta
            b = o < thresh3
            hough_vote[i][j] += np.sum(b & close_enough)

    return hough_vote

    # tlen = len(thetas)
    # clen = len(cs)
    # x, y = np.where(gradmag > thresh1)
    # votes = np.zeros((tlen, clen))
    # for i in range(0, tlen):
    #     for j in range(0, clen):
    #         dist = check_distance_from_line(y, x, thetas[i], cs[j], thresh2)
    #         orient = gradori[x, y] - thetas[i]
    #         bools = orient < thresh3
    #         votes[i][j] += np.sum(bools & dist)
    # return votes


# TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if:
# (a) Its votes are greater than thresh, **and**
# (b) Its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
# The input `nbhd` is an odd integer, and the nbhd x nbhd neighborhood is defined with the
# coordinate of the potential local maxima placing at the center.
# Return a list of (theta, c) pairs.
def localmax(votes, thetas, cs, thresh, nbhd):
    p = np.pad(votes, ((nbhd//2, nbhd//2), (nbhd//2, nbhd//2)))

    def is_local_max(i, j):
        val = votes[i, j]
        # for y in range(-nbhd // 2, nbhd // 2):
        #     for x in range(-nbhd // 2, nbhd // 2):
        #         if votes[i + x, j + y] > val:
        #             return False
        if val >= (p[i:i+nbhd, j:j+nbhd]).max():
            return True
        return False

    ans = []
    for i, theta in enumerate(thetas):
        for j, c in enumerate(cs):
            if votes[i, j] > thresh and is_local_max(i, j):
                ans.append([theta, c])
    return ans

    # mid = nbhd//2
    # pad = np.pad(votes, ((mid, mid), (mid, mid)))
    # result = []
    # for i in range(len(votes)):
    #     for j in range(len(votes[0])):
    #         padded = pad[i:i+nbhd, j:j+nbhd]
    #         if votes[i][j] > thresh and votes[i][j] >= padded.max():
    #             result.append((thetas[i], cs[j]))
    # return result


# Final product: Identify lines using the Hough transform
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
