import skimage
import numpy

from Rescaleintensity import RescaleIntensity
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = skimage.io.imread(
        "/Users/lingrajsvannur/Desktop/AutoML/cellprofilebuck/images/001001-2-001001003.tif")
    mask = image != 0

    print(image.shape)
    print(mask)

    input_image = {'pixel_data': image, "mask": mask, "multichannel": 0}
    rescaleobj = RescaleIntensity()
    a = rescaleobj.stretch(input_image)
