import cv2
import numpy as np


class ImageOrientation(object):
    def __init__(self):
        pass

    @staticmethod
    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (c_x, c_y) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        # compute the new bounding dimensions of the image
        n_w = int((h * sin) + (w * cos))
        n_h = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        matrix[0, 2] += (n_w / 2) - c_x
        matrix[1, 2] += (n_h / 2) - c_y
        # perform the actual rotation and return the image

        return cv2.warpAffine(image, matrix, (n_w, n_h))

    @staticmethod
    def get_centroids(contours):
        centroids = []
        for contour in contours:
            matrix = cv2.moments(contour)
            if matrix["m00"] != 0.0:
                c_x = int(matrix["m10"] / matrix["m00"])
                c_y = int(matrix["m01"] / matrix["m00"])
                centroids.append((c_x, c_y))

        return centroids

    def __call__(self, image, **kwargs):
        # convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # threshold the grayscale image
        ret, thresh = cv2.threshold(gray, 0, 255, 0)

        # find outer contour
        cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        # get rotated rectangle from outer contour
        rotrect = cv2.minAreaRect(cntrs[0])

        # get angle from rotated rectangle
        angle = rotrect[-1]

        # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)

        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle

        # find image centroid position and check if the image is inverted
        centroids = self.get_centroids(self.rotate_bound(image, angle))
        median_centroid = np.median(centroids, axis=0)
        if median_centroid[1] < image.shape[1]:
            angle = 180 + angle

        return self.rotate_bound(img, angle)


if __name__ == '__main__':
    # ===== MAIN =====
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_image', help='Input image name')
    parser.add_argument('output_image', help='Output image name')
    parser.add_argument('--show', help='Show result image', action='store_true')
    args = parser.parse_args()

    image_path = args.input_image
    if not os.path.exists(image_path):
        print("ERROR {} not found!".format(image_path))
        exit()

    img = cv2.imread(image_path)
    orient = ImageOrientation()
    output = orient(img)
    # write result to disk
    cv2.imwrite(args.output_image, output)

    # Show image
    if args.show:
        cv2.imshow(description, output)
        cv2.waitKey(0)
