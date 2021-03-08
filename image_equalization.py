class EqualizeImage(object):
    """
        Args:
            Image (Array): Array image of size (H, W, C).
        Returns:
            Tensor: Equalized image.
        Example:
            eq = EqualizeImage()
            new_image = eq(image)
        """
    def __init__(self):
        pass

    @staticmethod
    def white_balance(image):
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        return result

    @staticmethod
    def image_equalization(image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv = (255 * (img_yuv - np.min(img_yuv)) / np.max(img_yuv - np.min(img_yuv))).astype(np.uint8)
        # equalize the histogram of the Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        # convert the YUV image back to RGB format

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def __call__(self, image):
        final = 255*self.white_balance(image)

        return self.image_equalization(final)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_image', help='Input image path')
    parser.add_argument('output_image', help='Output image path')
    parser.add_argument('--show', help='Show result image', action='store_true')
    args = parser.parse_args()

    image_path = args.image_path
    if not os.path.exists(image_path):
        print("ERROR {} not found!".format(image_path))
        exit()

    tic = time()
    img = image = cv2.imread(file, 1).astype(np.float32) / 255
    eq = EqualizeImage()
    eq(img)
    print("Equalized image {0} in {1} seconds".format(args.input_image, (time() - tic)))
    output = cv2.imwrite(args.output_image, output)

    # Show image
    if args.show:
        cv2.imshow(description, stacked_image)
        cv2.waitKey(0)
