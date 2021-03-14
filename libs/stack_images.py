import argparse
import cv2
import os
import numpy as np
from time import time


class ImageStacker(object):
    """
    Args:
        Folder path: Folder of images of size (H, W, C) to stack.
    Returns:
        Image: Single stacked image of size (H, W, C).
    Example:
        stacker = ImageStacker()
        new_image = stacker.ecc(images_folder_path)
    """

    def __init__(self):
        self.first_image = None
        self.image_stacked = None

    def ecc(self, files_list):
        # Align and stack images with ECC method
        # Slower but more accurate
        warp_matrix = np.eye(3, 3, dtype=np.float32)

        _iter = 1000
        eps = 1e-10
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, _iter,  eps)

        for file in files_list:
            image = cv2.imread(file, 1).astype(np.float32) / 255
            print(file)
            if self.first_image is None:
                # convert to gray scale floating point image
                self.first_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.image_stacked = image
            else:
                # Estimate perspective transform
                (s, warp_matrix) = cv2.findTransformECC(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    self.first_image,
                    warp_matrix,
                    cv2.MOTION_HOMOGRAPHY,
                    criteria,
                    inputMask=None,
                    gaussFiltSize=1
                )
                w, h, _ = image.shape
                # Align image to first image
                image = cv2.warpPerspective(image, warp_matrix, (h, w))
                self.image_stacked += image

        self.image_stacked /= len(files_list)
        self.image_stacked = (self.image_stacked*255).astype(np.uint8)
        return self.image_stacked

    def orb(self, files_list):
        # Align and stack images by matching ORB key-points
        # Faster but less accurate
        orb = cv2.ORB_create()

        # disable OpenCL because of bug in ORB in OpenCV 3.1
        cv2.ocl.setUseOpenCL(False)

        first_kp = None
        first_des = None
        for file in files_list:
            print(file)
            image = cv2.imread(file, 1)

            image_f = image.astype(np.float32) / 255

            # compute the descriptors with ORB
            kp = orb.detect(image, None)
            kp, des = orb.compute(image, kp)

            # create BFMatcher object
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            if self.first_image is None:
                # Save key-points for first image
                self.image_stacked = image_f
                self.first_image = image
                first_kp = kp
                first_des = des
            else:
                # Find matches and sort them in the order of their distance
                matches = matcher.match(first_des, des)
                matches = sorted(matches, key=lambda x: x.distance)

                src_pts = np.float32(
                    [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Estimate perspective transformation
                warp_matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                w, h, _ = image_f.shape
                image_f = cv2.warpPerspective(image_f, warp_matrix, (h, w))
                self.image_stacked += image_f

        self.image_stacked /= len(files_list)
        self.image_stacked = (self.image_stacked*255).astype(np.uint8)
        return self.image_stacked


if __name__ == '__main__':
    # ===== MAIN =====
    # Read all files in directory

    stacked_image = None
    description = None

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', help='Input directory of images ()')
    parser.add_argument('output_image', help='Output image name')
    parser.add_argument('--method', help='Stacking method ORB (faster) or ECC (more precise)')
    parser.add_argument('--show', help='Show result image', action='store_true')
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print("ERROR {} not found!".format(image_folder))
        exit()

    file_list = os.listdir(image_folder)
    file_list = [os.path.join(image_folder, x)
                 for x in file_list if x.endswith(('.jpg', '.png', '.bmp'))]

    if args.method is not None:
        method = str(args.method)
    else:
        method = 'KP'

    tic = time()

    stacker = ImageStacker()

    if method == 'ECC':
        # Stack images using ECC method
        description = "Stacking images using ECC method"
        print(description)
        stacked_image = stacker.ecc(file_list)

    elif method == 'ORB':
        # Stack images using ORB keypoint method
        description = "Stacking images using ORB method"
        print(description)
        stacked_image = stacker.orb(file_list)

    else:
        print("ERROR: method {} not found!".format(method))
        exit()

    print("Stacked {0} in {1} seconds".format(len(file_list), (time()-tic)))

    print("Saved {}".format(args.output_image))
    cv2.imwrite(str(args.output_image), stacked_image)

    # Show image
    if args.show:
        cv2.imshow(description, stacked_image)
        cv2.waitKey(0)
