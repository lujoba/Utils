import argparse
import cv2
import os
import numpy as np


class BandPassFilter(object):
    """
    Args:
        Image (Array): Monochromatic array image of size (H, W).
    Returns:
        Tensor: Contrast enhanced image.
    Example:
        bpf = BandPassFilter(r_out=500, r_in=50)
        new_image = bpf(image)
    """
    def __init__(self, r_out=500, r_in=50):
        self.r_out = r_out
        self.r_in = r_in

    @staticmethod
    def replace_zeroes(data):
        # Find and replaces the zeroes of an array
        min_nonzero = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    def __call__(self, image):
        # A Band Pass Filter improves the edge contrast
        # of the image
        rows, cols = image.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.zeros((rows, cols, 2), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]

        mask_area = np.logical_and(
            ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= self.r_in ** 2),
            ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= self.r_out ** 2)
        )
        mask[mask_area] = 1

        # apply mask and inverse DFT
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift * mask

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)

        return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_image', help='Input images path')
    parser.add_argument('output_image', help='Output images path')
    parser.add_argument('--show', help='Show result image', action='store_true')
    args = parser.parse_args()

    file = args.input_image
    if not os.path.exists(file):
        print("ERROR {} not found!".format(file))
        exit()

    img = cv2.imread(file, 0)
    bpf = BandPassFilter(r_out=500, r_in=50)
    output = bpf(img)
    cv2.imwrite(args.output_image, output)

    # Show image
    if args.show:
        cv2.imshow(description, stacked_image)
        cv2.waitKey(0)
