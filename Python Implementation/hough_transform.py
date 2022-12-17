import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import image
import matplotlib

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def crop_resizeimage(im,imbw):
    meso = (cords[:, 0] + cords[:, 1]) / 2
    linewidth = -40
    # meanval = np.mean(meso)
    minval = int(np.min(meso)) + linewidth
    maxval = int(np.max(meso)) - linewidth

    newimage = im[:, minval:maxval, :]
    newimagebw = imbw[:, minval:maxval]
    cropresized= resize(newimage, (416, 416))
    cropresizedbw= resize(newimagebw, (416, 416))
    return cropresized,cropresizedbw

def houghtransform(image):
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi , np.pi, 500)
    h, theta, d = hough_line(image, theta=tested_angles)

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    cords=[]
    # cords.clear()

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        # y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        xx0 = (dist - 0 * np.sin(angle)) / np.cos(angle)
        xx1 = (dist - image.shape[0] * np.sin(angle)) / np.cos(angle)
        y0=0
        y1=image.shape[0]
        origin = np.array([xx0, xx1])

        ax[2].plot(origin, [y0, y1], '-r')
        cords.append([xx0, xx1,y0, y1, np.cos(angle)/np.sin(angle)])

    ax[2].set_xlim((0,image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')
    plt.tight_layout()
    plt.show()
    return cords

im = image.imread('/Users/charalamposp/Desktop/test.jpg')
print(im.dtype)
print(im.shape)
imgray = rgb2gray(im)

imbw=imgray > 255/10
# plt.imshow(imbw, cmap='gray', interpolation='nearest')

cords=np.array(houghtransform(imbw))



