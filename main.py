import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os import mkdir, path

in_folder, ext = 'input', 'tif'
out_folder = 'NikitaNigmatullinOutputs'


def IFFT(arr):
    ''' Recursive Inverse Fast Fourier Transformation implementation 
    '
    '   Arguments: arr - 1d numpy array
    '   Return: 1d numpy array
    '''
    return FFT(arr, inverse=True) / arr.shape[0]


def FFT(arr, inverse=False):
    ''' Recursive Fast Fourier Transformation implementation
    '
    '   Arguments: arr - 1d numpy array, inverse - Boolean
    '   Return: 1d numpy array
    '''
    sign = 1 if inverse else -1

    if arr.shape[0] == 1:
        # Return the array of lenght 1
        return arr
    else:
        # Recursively run FFT for even and odd arr's elements
        even, odd = FFT(arr[0::2], inverse=inverse), FFT(
            arr[1::2], inverse=inverse)

        # Calculate the omega value
        omega = sign * 2j * np.pi / arr.shape[0]

        # Calculate the range of values
        values = np.exp(omega * np.arange(arr.shape[0] // 2)) * odd

        return np.concatenate([even + values, even - values])


def compress(img):
    ''' Compress img using FFT and IFFT
    '
    '   Arguments: img - 2d numpy array
    '   Return: 2d numpy array
    '''
    # Do 2D FFT
    img = np.apply_along_axis(FFT, 1, img)
    img = np.apply_along_axis(FFT, 0, img)

    # Drop some values
    img[img < np.mean(img) - 13 * np.std(img)] = 0

    # Do 2D IFFT
    img = np.apply_along_axis(IFFT, 1, img)
    img = np.apply_along_axis(IFFT, 0, img)

    # Return real values of img
    return np.real(img)


def save_image(path, img):
    ''' Save image
    '
    '   Arguments: path - string, img - 2d numpy array
    '   Return: None
    '''
    # Get the file name
    f_name = path.split('/')[1]
    # Get the name of the file without extension
    name = f_name.split('.')[0]

    # Map the image to grayscale format and save it in format `nameCompressed.ext`
    mpimg.imsave(f'{out_folder}/{name}Compressed.{ext}',
                 img, format='TIFF', cmap='gray')


if __name__ == "__main__":
    # Get filenames from input/ with *.tif extension
    f_names = glob(f'{in_folder}/*.{ext}')
    # Read all files as images
    imgs = [mpimg.imread(f) for f in f_names]

    # Compress images using FFT
    compressed_imgs = list(map(compress, imgs))

    # Create folder for output images
    if not path.exists(out_folder) or path.isfile(out_folder):
        mkdir(out_folder)

    # Save images
    [save_image(path, img) for path, img in zip(f_names, compressed_imgs)]
