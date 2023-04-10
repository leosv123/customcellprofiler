from MeasureObjectIntensity import MeasureObjectIntensity
from Rescaleintensity import RescaleIntensity
from MeasureObjectSizeShape import MeasureObjects
from IdentifyPrimaryObjectsfile import IdentifyPrimaryObject
import skimage
import numpy
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from argparse import ArgumentParser
import math
import warnings
warnings.filterwarnings('ignore')


def resize(image_pixels, mask, resize_method, interpolation, resizing_factor_x=0.25, resizing_factor_y=0.25):

    shape = numpy.array(image_pixels.shape).astype(float)

    if resize_method == "Resize by a fraction":

        factor_x = resizing_factor_x
        factor_y = resizing_factor_y

        height, width = shape[:2]

        height = numpy.round(height * factor_y)
        width = numpy.round(width * factor_x)
        new_shape = []
        new_shape += [height, width]
        new_shape = numpy.asarray(new_shape)

        if interpolation == "Nearest Neighbor":
            order = 0

        elif interpolation == "Bilinear":
            order = 1

        output_pixels = skimage.transform.resize(
            image_pixels, new_shape, order=order, mode="symmetric")
        mask = skimage.transform.resize(
            mask, new_shape, order=0, mode="constant")
        resized_mask = skimage.img_as_bool(mask)
    return output_pixels, resized_mask


if __name__ == "__main__":
    image_path = "/Users/lingrajsvannur/Downloads/masks/001001-1-001001003.tif"
    image = skimage.io.imread(image_path)
    mask = image != 1

    input_image = {'pixel_data': image, "mask": mask,
                   "multichannel": 0, "image_name": image_path.split('/')[-1]}
    rescaleobj = RescaleIntensity()
    rescaled_intensity_image = rescaleobj.stretch(input_image)
    image_pixels = rescaled_intensity_image
    resize_method = "Resize by a fraction"
    interpolation = "Bilinear"
    resizing_factor_x = 1.5
    resizing_factor_y = 1.5
    resized_image, resized_mask = resize(
        image_pixels, mask, resize_method, interpolation, resizing_factor_x, resizing_factor_y)
    print("Resulting image after resizing is of shape: ", resized_image.shape)
    print("Resulting Mask after resizing is of shape: ", resized_image.shape)
    skimage.io.imsave("resized_image.png", resized_image)
    configFile = OmegaConf.load(
        "/Users/lingrajsvannur/Desktop/AutoML/cellprofilebuck/identifyprimaryobj.yaml")
    po = IdentifyPrimaryObject(config_data=configFile)
    po.read_image_fn()
    binary_image, global_threshold, sigma, add_fg_bg_measurements_dict = po.threshold_fn()
    finalstats, labeled_image, outline_image, objects = po.fill_holes_fn(
        binary_image, global_threshold)
    print(finalstats)
    desired_properties = [
        "label",
        "image",
        "area",
        "perimeter",
        "bbox",
        "bbox_area",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "centroid",
        "equivalent_diameter",
        "extent",
        "eccentricity",
        "convex_area",
        "solidity",
        "euler_number",
    ]

    a = MeasureObjects("yeast", calculate_zernikes=False)
    ans, fullresult = a.analyze_objects(objects, desired_properties)
    MOIntensity = MeasureObjectIntensity(input_images=[input_image], objects_list=[
                                         objects], object_name="yeast")
    intensity_measurements, stats = MOIntensity.calculate()
