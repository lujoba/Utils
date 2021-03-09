# Image Utils

A library collection of OpenCV wrappers for image processing

## Dependencies

- [opencv](https://opencv.org/): Image processing library;
- [numpy](https://numpy.org/): The fundamental package for scientific computing with Python.

## Scripts

- [Band Pass Filter](https://github.com/lujoba/utils/blob/main/band_pass_filter.py): Script for enhancement of image's edges;
- [Image Equalization](https://github.com/lujoba/Utils/blob/main/image_equalization.py): Correct the white balance and equalize the light of an image;
- [Stacking images](https://github.com/lujoba/utils/blob/main/stack_images.py): Image stacking script for pictures of the same subject. The resulting image will have a higher pixel density.

## Set Up

1. Check out the code;
2. Install requirements:
    ```
    pipenv install
    ```
3. Use as CLI, as the example below, or call the class in your project:
    ```
   pipenv python -m stack_images.py folder/of/input/images path/of/output/images --method ORB 
    ```
4. Cite me;
5. Have fun.
