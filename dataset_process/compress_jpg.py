import cv2
from imutils import paths
import os

folder_with_imgs = "atlantic1"

folder_with_compressed = folder_with_imgs + "_compressed"

list_imgs = list(paths.list_images(os.path.join(os.getcwd(), folder_with_imgs)))

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

if not os.path.exists(os.path.join(os.getcwd(), folder_with_compressed)):
    os.makedirs(os.path.join(os.getcwd(), folder_with_compressed))

counter = 0
for img in list_imgs:
    img_name = os.path.split(img)[1]
    read_img = cv2.imread(img)
    # result, encimg = cv2.imencode('.jpg', read_img, encode_param)
    # to decode use this
    # decimg = cv2.imdecode(encimg, 1)
    # if result:
    cv2.imwrite(os.path.join(os.getcwd(), folder_with_compressed, img_name), read_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    counter += 1
    print("Encoded {}/{}".format(counter, len(list_imgs)), end='\r')
print("Finished! Check folder {}".format(os.path.join(os.getcwd(), folder_with_compressed)))