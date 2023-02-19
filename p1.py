import numpy as np
from PIL import Image

############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    img = Image.open(filename)
    img_array = np.array(img)
    img_array = img_array.astype(np.float)
    img_array /= 255
    return img_array

### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):
    m = len(img) #image rows
    n = len(img[0]) #image columns

    l = len(filt) #filter rows
    k = len(filt[0]) #filter cols

    init = np.zeros((m, n)) if isinstance(img[0][0], list) else np.zeros((m,n,3))
    
    def does_fit(i, j):
        if not (0 <= i - l //2 < m and 0 <= i + l // 2 < m):
            return False
        if not (0 <= j - k // 2 < n and 0 <= j + k // 2 < n):
            return False
        return True
    
    def filter(i, j):
        if not does_fit(i, j):
            return img[i][j]
        ans = 0. if isinstance(img[0][0], list) else np.zeros((3))
        for x in range(-l // 2 + 1, l // 2 + 1):
            for y in range(-k // 2 + 1, k // 2 + 1):
                fx, fy = x + l // 2, y + k // 2
                ans += filt[fx][fy] * img[i + x][j + y]
        return ans
    
    for i in range(m):
        for j in range(n):
            init[i][j] = filter(i, j)
    
    return init

        

### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    def gaussian(x, y):
        const = 1 / (2 * np.pi * sigma ** 2)
        exponent = - (x ** 2 + y ** 2) / (2 * sigma ** 2)
        return const * np.exp(exponent)
    
    filter = np.zeros((k, k))

    for x in range(-k // 2, k // 2 + 1):
        for y in range(-k // 2, k // 2 + 1):
            i = x + k // 2
            j = y + k // 2
            
            filter[i][j] = gaussian(i, j)
    filter /= np.sum(filter)
    
    return filter


### TODO 4: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. 
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
### convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    m = len(img)
    n = len(img[0])

    grayscale = [img[i][:] for i in range(m)] #make deep copy

    for i in range(m):
        for j in range(n):
            r, g, b = grayscale[i][j]

            grayscale[i][j] = 0.2125 * r + 0.7154 * g + 0.0721 * b 
    
    filt = gaussian_filter(5, 1)
    grayscale = convolve(grayscale, filt)
    x_gradient = convolve(grayscale, [0.5, 0, -0.5])
    y_gradient = convolve(grayscale, [[0.5], [0], [-0.5]])
    
    image_gradient = np.zeros((n,m))
    image_orientation = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            image_gradient[i][j] = np.sqrt(x_gradient[i][j] ** 2 + y_gradient[i][j] ** 2)
            image_orientation[i][j] = np.arctan2(y_gradient[i][j], x_gradient[i][j])
    return image_gradient, image_orientation
    
##########----------------Line detection----------------

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are numpy arrays of the same shape, representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    n = len(x)
    is_less = [False] * n
    for i in range(n):
        pos_x, pos_y = x[i], y[i]
        d = pos_x * np.cos(theta) + y * np.sin(theta) + c 
        is_less[i] = d < thresh 
    return is_less


### TODO 6: Write a function to draw a set of lines on the image. 
### The `img` input is a numpy array of shape (m x n x 3).
### The `lines` input is a list of (theta, c) pairs. 
### Mark the pixels that are less than `thresh` units away from the line with red color,
### and return a copy of the `img` with lines.
def draw_lines(img, lines, thresh):
    n = len(img)
    m = len(img[0])

    result_img = [[img[i][j][:] for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            result_img[i][j] = [255, 0, 0] if any([check_distance_from_line(i, j, line, thresh) for line in lines]) else result_img[i][j]
    return result_img


### TODO 7: Do Hough voting. You get as input the gradient magnitude (m x n) and the gradient orientation (m x n), 
### as well as a set of possible theta values and a set of possible c values. 
### If there are T entries in thetas and C entries in cs, the output should be a T x C array. 
### Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1, **and** 
### (b) Its distance from the (theta, c) line is less than thresh2, **and**
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    pass   

### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if: 
### (a) Its votes are greater than thresh, **and** 
### (b) Its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
### The input `nbhd` is an odd integer, and the nbhd x nbhd neighborhood is defined with the 
### coordinate of the potential local maxima placing at the center.
### Return a list of (theta, c) pairs.
def localmax(votes, thetas, cs, thresh, nbhd):
    pass
  
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
   
    
