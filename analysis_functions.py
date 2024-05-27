# analysis_functions.py
#
# modules: scikit-learn, cv2 (OpenCV), NumPy, math, matplotlib
#
#
# lines are defined as a tuple, (top, bottom, point), where top/bottom = slope

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import cv2
import math


def canny(img, lower=100, upper=200, ap=5):
    im_canny = cv2.Canny(img, lower, upper, apertureSize=ap)
    return im_canny

def show(img):
    # Display the image.
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(12, 3))

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_axis_off()

    # Display the histogram.
    ax2.hist(img.ravel(), lw=0, bins=256)
    ax2.set_xlim(0, img.max())
    ax2.set_yticks([])

    plt.show()

def display_image(data):
    from matplotlib import pyplot as plt
    plt.imshow((data), interpolation='nearest')
    plt.show()

def compare(im1, im2):
    from matplotlib import pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow((im1))
    ax2.imshow((im2))
    f.show()
    plt.show()

def im_to_array(image):
    from PIL import Image
    import numpy as np
    I = Image.open(image)
    I = np.array(I)
    return I

def arr_to_grayscale(arr):
    if len(np.shape(arr)) == 2:
        return np.float64(arr)
    im = np.uint8(arr)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return np.float64(im)

def sobel(im):
    a = np.uint8(im)
    ksize = 3
    gX = cv2.Sobel(a, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(a, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    return combined

def remove_noise(im, num_regions=2, x=10):
    square_size = 2 * x + 1
    a = np.array(im)
    r, c = a.shape
    info_arr = np.zeros((im.size, 1))
    s1, s2 = im.shape
    counter = 0
    for i in range(0, r):
        for j in range(0, c):
            i_min = i-x
            i_max = i_min+square_size
            i_min = max(0, i_min)
            i_max = min(r-1, i_max)

            j_min = j-x
            j_max = j_min+square_size
            j_min = max(0, j_min)
            j_max = min(c-1, j_max)

            filter_arr =  a[i_min:i_max, j_min:j_max]
            number = 0
            s = np.sum(filter_arr == 0)
            for k in range(1, num_regions):
                t = np.sum(filter_arr == k)
                if t > s:
                    s = t
                    number = k
            info_arr[counter, 0] = number
            counter += 1
    
    return np.reshape(info_arr, im.shape)

def edge_std(im, x=100):
    square_size = x
    a = np.array(im)
    a = cv2.GaussianBlur(np.uint8(a), (5, 5), 0)
    a = np.float64(a)
    r, c = a.shape
    info_arr = np.zeros((im.size, 1))
    s1, s2 = im.shape
    counter = 0
    i = 0
    j = 0
    while True:
        i_min = i
        i_max = i_min+square_size
        i_min = max(0, i_min)
        i_max = min(r, i_max)
        j = 0
        while True:
            j_min = j
            j_max = j_min+square_size
            j_min = max(0, j_min)
            j_max = min(c, j_max)

            filter_arr = im[i_min:i_max, j_min:j_max]
            a[i_min:i_max, j_min:j_max] = np.std(filter_arr)
            j += square_size
            if j_max == c:
                break
        i += square_size
        if i_max == r:
            break
    return a

def find_intercept(slope, point):
    y = point[1]
    x = point[0]
    c = y - slope*x
    return c

def find_endpoints(top, bottom, point, x_max, y_max):
    if top*bottom < 0:
        bottom = abs(bottom)
        top = -1 * abs(top)
    x_max = max(2*point[0] + 10, x_max)
    if bottom != 0:
        step_max = int((x_max - point[0]) / bottom) + 1
        delta_x = step_max * bottom
        delta_y = step_max * top
    else:
        delta_x = 0
        delta_y = y_max
    f1 = (point[0] + delta_x, point[1] + delta_y)
    f2 = (point[0] - delta_x, point[1] - delta_y)

    if point[0] == 0:
        g1 = point
        g2 = f1
    else:
        g1 = point
        g2 = f2
        if g2[1] < 0:
            g2 = f1
    return f1, f2


def get_line(im, point, top, bottom, thickness=1):
    a = arr_to_grayscale(im)
    r, c = a.shape
    zero_im = np.zeros(a.shape)
    p1, p2 = find_endpoints(top, bottom, point, c, r)
    zero_im = cv2.line(zero_im, p1, p2, 1, thickness)
    return zero_im != 0

def draw_line(im, point, top, bottom, thickness=1, color=(0, 0, 255)): # im needs to be an RGB image
    zero_im = get_line(arr_to_grayscale(im), point, top, bottom, thickness=thickness)
    a = np.uint8(im)
    a = a[:, :, 0:3]
    a[:, :, 0][zero_im] = color[0]
    a[:, :, 1][zero_im] = color[1]
    a[:, :, 2][zero_im] = color[2]
    return np.uint8(a)

def gray_to_RGB(im):
    a = im * (255 / np.max(im))
    a = np.uint8(im)
    a = cv2.cvtColor(a,cv2.COLOR_GRAY2RGB)
    return a
 
def find_lines(im, return_point=False): # finds the first main line
    lines = cv2.HoughLines(np.uint8(im),1,np.pi/180,200)
    counter = 0
    for rho, theta in lines[0]:
        a1 = np.cos(theta)
        b1 = np.sin(theta)
        x0 = a1*rho
        y0 = b1*rho
        x1 = int(x0 + 1000*(-b1))
        y1 = int(y0 + 1000*(a1))
        x2 = int(x0 - 1000*(-b1))
        y2 = int(y0 - 1000*(a1))
        top = y2-y1
        bottom = x2-x1
        if top * bottom < 0:
            top = -1 * abs(top)
            bottom = abs(bottom)
        point = (x1, y1)
        counter += 1
    if return_point:
        return top, bottom, point
    return top, bottom

def line_angle(top, bottom):
    if bottom == 0:
        slope = math.inf
    else:
        slope = top/bottom
    angle = math.degrees(math.atan(slope))
    return angle

def angle_to_slope(angle): # angle is in degrees, not radians
    a = math.radians(angle)
    b = math.cos(a)
    t = math.sin(a)
    if t*b < 0:
        t = -1 * abs(t)
        b = abs(b)
    return t, b

def line_average(lines): # lines is a list of lines; the argument point will lie on the new averaged line
    avg_angle = 0
    for line in lines:
        t, b, p = line
        angle = line_angle(t, b)
        avg_angle += angle

    avg_angle /= len(lines)
    if avg_angle == 90 or avg_angle == 270:
        t = 1
        b = 0
        return t, b
    if avg_angle == 0 or avg_angle == 180:
        t = 0
        b = 1
        return t, b
    slope = math.tan(math.radians(avg_angle))
    num_digits = int(math.log(abs(slope)) / math.log(10)) + 1
    if num_digits > 3:
        factor = 1
    else:
        factor = 10 ** (3 - num_digits)
    slope *= factor
    return round(slope), factor

def line_intersection(a, line_1, line_2, thickness=1): # line is a tuple (top, bottom, point)
    t1, b1, p1 = line_1
    t2, b2, p2 = line_2
    z1 = 1 * get_line(a, p1, t1, b1, thickness=thickness)
    z2 = 1 * get_line(a, p2, t2, b2, thickness=thickness)
    if np.any((z1 + z2) == 2):
        return True
    return False

def line_distance(line_1, line_2): # line is a tuple (top, bottom, point), also assumes both lines are parallel
    t1, b1, p1 = line_1
    t2, b2, p2 = line_2
    if b1 == 0:
        return abs(p1[0] - p2[0])
    m = t1 / b1
    angle = math.atan(-m)
    c1 = find_intercept(m, p1)
    c2 = find_intercept(m, p2)
    return abs(c1 - c2) * math.cos(angle)

def remove_intersecting_lines(a, main_angle, lis, thickness=10):
    lis_2 = []
    while True:
        line = lis[0]
        best_line = line
        lis.remove(line)
        for j in range(0, len(lis)):
            line_2 = lis[j]
            distance = line_distance(best_line, line_2)
            max_distance = 50
            if line_intersection(a, best_line, line_2, thickness=thickness) or distance < max_distance:
                t1, b1, p1 = best_line
                t2, b2, p2 = line_2
                best_angle = line_angle(t1, b1)
                angle_2 = line_angle(t2, b2)
                if abs(angle_2 - main_angle) < abs(best_angle - main_angle):
                    best_line = line_2
                lis.remove(line_2)
            if j >= len(lis)-1:
                break
        lis_2.append(best_line)
        if len(lis) == 0:
            break
    return lis_2

def generate_gridlines(im, line, distance, thickness=10): # writes equally spaced parallel lines on 2D array, around the argument 'line'
    points = get_equidistant_points(im, line, distance)
    t, b, p = line
    a = np.zeros(np.shape(im))
    for point in points:
        p1, p2 = find_endpoints(t, b, point, im.shape[1], im.shape[0])
        a = cv2.line(a, p1, p2, 1, thickness=thickness)
    return a != 0

def keep_lines(edge_map):
    lines = cv2.HoughLinesP(np.uint8(edge_map),1,np.pi/180,100,minLineLength=50,maxLineGap=10)
    lis = []
    thickness=3
    zero_im = np.zeros(np.shape(edge_map))
    r, c = np.shape(edge_map)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        t = y2-y1
        b = x2-x1
        p1 = (x1, y1)
        p2 = (x2, y2)
        zero_im = cv2.line(zero_im, p1, p2, 1, thickness)
    zero_im = zero_im != 0
    return np.uint8(zero_im)

def check_line(im, line):
    r, c = im.shape
    t, b, point = line
    x = point[0]
    y = point[1]
    if t == 0:
        if y < r and y >= 0:
            return True
    elif b == 0:
        if x < c and x >= 0:
            return True
    else:
        slope = t/b
        y_intercept = find_intercept(slope, point)
        if y_intercept < r and y_intercept >= 0:
            return True
        x_intercept = -y_intercept / slope
        if x_intercept < c and x_intercept >= 0:
            return True
        y_right = slope * (c-1) + y_intercept # this is the right side of the image, x = c-1
        if y_right < r and y_right >= 0:
            return True
        x_bottom = ((r-1) - y_intercept) / slope # this is the bottom edge of the image, y = r-1
        if x_bottom < c and x_bottom >= 0:
            return True
    return False

def get_equidistant_points(im, line, distance): # line is a tuple (top, bottom, point); top/bottom = slope; point is a point on the line
    t, b, point = line
    p_x = point[0]
    p_y = point[1]
    r, c = im.shape
    lis = [point]
    d = (t*t + b*b) ** 0.5
    delta_y = -b * abs(distance) / d
    delta_x = t * abs(distance) / d
    i = 1
    while True:
        k = 0
        p1 = (round(p_x + i * delta_x), round(p_y + i * delta_y))
        p2 = (round(p_x - i * delta_x), round(p_y - i * delta_y))
        if check_line(im, (t, b, p1)):
            lis.append(p1)
            k = 1
        if check_line(im, (t, b, p2)):
            lis.append(p2)
            k = 1
        if k == 0:
            break
        i += 1
    return lis

def find_main_lines(im_RGB, edge_map, color=(0,255,255)):
    im = arr_to_grayscale(im_RGB)
    main_t, main_b, main_point = find_lines(edge_map, return_point=True)
    main_line = (main_t, main_b, main_point)
    main_angle = line_angle(main_t, main_b)
    lines = cv2.HoughLinesP(np.uint8(edge_map),1,np.pi/180,100,minLineLength=30,maxLineGap=10)
    lis = []
    d_lis = []
    lengths = []
    zero_im = np.zeros(np.shape(im))
    for line in lines:
        x1,y1,x2,y2 = line[0]
        t = y2-y1
        b = x2-x1
        p1 = (x1, y1)
        p2 = (x2, y2)
        l = (t, b, p1)
        line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        angle = line_angle(t, b)
        d2 = line_distance(main_line, l)
        if abs(angle - main_angle) < 10 and d2 != 0:
            zero_im = cv2.line(zero_im, p1, p2, 1, 3)
            n = len(lis)
            for i in range(0, len(lis)):
                l1 = lis[i]
                d1 = line_distance(main_line, l1)
                if d2 < d1:
                    lis.insert(i, l)
                    lengths.insert(i, line_length)
                    d_lis.insert(i, d2)
                    break
            if len(lis) == n:
                lis.append(l)
                lengths.append(line_length)
                d_lis.append(d2)
    distance = 30
    i = 0
    while True:
        current_set = [main_line]
        equal_points = get_equidistant_points(im, main_line, distance)
        if len(equal_points) == 1:
            break
        score = 0
        for j in range(1, len(equal_points)):
            min_dist = 0
            max_dist = 0.2 * distance
            p = equal_points[j]
            all_lines = []
            current_line = (main_t, main_b, p)
            break_dist = (max_dist) + line_distance(main_line, current_line)
            line_exists = False
            for k in range(0, len(lis)):
                line_2 = lis[k]
                current_dist = line_distance(current_line, line_2)
                current_length = lengths[k]
                if current_dist > min_dist:
                    if current_dist < max_dist:
                        if not line_exists:
                            best_line = line_2
                            closest_line = line_2
                            best_distance = current_dist
                            best_length = current_length
                            line_exists = True
                        elif current_length > best_length:
                            best_line = line_2
                            best_length = current_dist
                        if current_dist < best_distance:
                            best_distance = current_dist
                            closest_line = line_2
                        all_lines.append(line_2)
                    elif d_lis[k] > break_dist:
                        break
            if line_exists:
                all_lines.append(current_line)
                bt, bb = line_average(all_lines)
                best_line = (bt, bb, closest_line[2])
                score += 1
            else:
                best_line = current_line
            current_set.append(best_line)
        score = score / len(current_set)
        if distance < 30:
            score = 0
        if i == 0:
            best_score = score
            best_set = current_set
            best_equal_points = equal_points
        elif score > best_score:
            best_score = score
            best_set = current_set
            best_equal_points = equal_points
        distance += 1
        i += 1

    im_copy = np.array(im_RGB)
    for found_line in best_set:
        t, b, point = found_line
        im_copy = draw_line(im_copy, point, t, b, thickness=10, color=color)
    return im_copy

def image_parse(image_name):
    im_original = im_to_array(image_name)
    im = arr_to_grayscale(im_original)
    return np.uint8(im)