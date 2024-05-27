import cv2
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from analysis_functions import *
import sys
from PIL import Image
import random
import skimage.exposure as skie
import sys

#r = input("Number: ")

image_name = "/Users/anigasan/Desktop/cs156bstuff/CS156b/train/pid01457/study1/view1_frontal.jpg"
output_file = "/Users/anigasan/Desktop/cs156bstuff/CS156b/I1.jpg"

im_original = im_to_array(image_name)
im = arr_to_grayscale(im_original)
im = np.uint8(im)

#im = im[317:]
#im = cv2.resize(im, (256, 256))
im_o = np.array(im)
im = sobel(im)

clahe = cv2.createCLAHE(clipLimit=80, tileGridSize=(10, 10))
final_img = clahe.apply(im_o) + 30
im2 = sobel(final_img)

compare(im_o, final_img)
compare(im, im2)
#display_image(im)
sys.exit()


#show(im)

#im = skie.rescale_intensity(im, in_range=(0.4*255, 0.95*255), out_range=(0, 1))

#display_image(im)
#sys.exit()

num_clusters=150
km = KMeans(n_clusters=num_clusters)

km.fit(np.reshape(im, (im.size, 1)))

h2 = np.reshape(km.labels_, im.shape)
compare(h2, remove_noise(h2, num_regions=num_clusters, x=3))
h2 = remove_noise(h2, num_regions=num_clusters, x=3)
h2 = np.uint8(h2)

display_image(h2)

#h3 = 1 * (sobel(h2) != 0)
#h3 = np.uint8(h3)
#h3 = keep_lines(h3)


#display_image(edge_std(h2, x=10))
#im2 = canny(h2, lower=0, upper=3)
im2 = sobel(h2)
#compare(im2, im)
compare(im2, im_o)
#display_image(canny(h2, lower=0, upper=3))

#im_equal = find_main_lines(im_original, h3)
#im_equal = Image.fromarray((im_equal), 'RGB')
#im_equal.save(output_file)