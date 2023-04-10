import numpy as np
import skimage.io
import centrosome
import scipy


def get_threshold_manual(input_image, manual_threshold, automatic=False):
    """
    get the threshold value from the input
    """
    image_data = input_image["pixel_data"]
    return manual_threshold, manual_threshold, None


def apply_threshold(input_image, threshold_smoothing_scale, threshold, automatic):
    """
    apply the threshold to the image and smoothing
    """
    data = input_image["pixel_data"]
    mask = input_image["mask"]

    if automatic:
        sigma = 1
    else:
        sigma = threshold_smoothing_scale / 0.6744 / 2.0

    blurred_image = centrosome.smooth.smooth_with_function_and_mask(
        data,
        lambda x: scipy.ndimage.gaussian_filter(
            x, sigma, mode="constant", cval=0),
        mask,
    )

    return (blurred_image >= threshold) & mask, sigma


def add_fg_bg_measurements(object_name, input_image, binary_image):
    """
    Calculate the WeightVariance, SumOfEntropies
    """
    data = input_image["pixel_data"]

    mask = input_image["mask"]

    wv = centrosome.threshold.weighted_variance(data, mask, binary_image)

    entropies = centrosome.threshold.sum_of_entropies(data, mask, binary_image)

    wv_key = "Threshold_WeightedVariance_"+object_name
    wv_values = np.array([wv], dtype=float)
    entropy_key = "Threshold_SumOfEntropies_"+object_name
    entropy_values = np.array([entropies], dtype=float)

    return {wv_key: wv_values, entropy_key: entropy_values}
