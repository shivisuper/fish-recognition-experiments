# import the necessary packages
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        """
        dim = None
        (h, w) = image.shape[:2]
        if self.width is None and self.height is None:
            return image
        elif self.height is None:
            r = self.width / float(w)
            dim = (self.width, int(h * r))
        else:
            r = self.height / float(h)
            dim = (int(w * r), self.height)
        """
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)

