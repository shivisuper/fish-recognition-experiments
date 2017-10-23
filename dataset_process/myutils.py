import numpy as np
import cv2


def translate(image, x, y):
    """
    Translate the image by x and y pixels
    :param image: The image to be translated
    :type image: Image
    :param x: Translation along x-axis. Positive value means Right. Negative
    value means Left
    :type x: Integer
    :param y: Translation along y-axis. Positive value means Down. Negative
    value means Up
    :type y: Integer
    :return: Image
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def rotate(image, angle, center=None, scale=1.0):
    """
    Rotate the image by an angle, and perform any scaling (if specified)
    :param image: Image to be rotated
    :type image: Image
    :param center: (x,y) coordinate for the center around which rotation
    will be performed
    :type center: Integer
    :param angle: Angle in degrees (clockwise if negative, otherwise
    counter-clockwise)
    :type angle: Integer
    :param scale: By what factor the scaling should be performed. Default=1.0
    :type scale: Float
    :return: Image
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))


def resize(image, width=None, height=None, inter=cv2.INTER_CUBIC):
    """
    Resize the image on the basis of width and height of the image
    :param image: Image to be resized
    :param width: Width of the image in pixels
    :param height: Height of the image in pixels
    :param inter: The algorithm that will be used to perform resizing.
    Possible choices - INTER_CUBIC, INTER_LINEAR, INTER_NEAREST
    :return: Resized image
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    elif height is None and width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    elif width is None and height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        dim = (width, height)
    return cv2.resize(image, dim, interpolation=inter)
