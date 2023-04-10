import numpy as np
import scipy
import centrosome
import skimage


def separate_neighboring_objects(input_image, size_range, automatic_smoothing, low_res_maxima, automatic_suppression, labeled_image,
                                 unclump_method, fill_holes, object_count, watershed_method, smoothing_filter_size=None, maxima_suppression_size=None):
    """
    Separate objects based on local maxima or distance transform

    args:
        labeled_image: image labeled by scipy.ndimage.label
        object_count:  # of objects in image

    returns:  revised labeled_image, object count, maxima_suppression_size,
              LoG threshold and filter diameter
    """
    image = input_image["pixel_data"]
    mask = input_image["mask"]
    blurred_image = smooth_image(
        image, mask, automatic_smoothing, size_range, smoothing_filter_size)

    if size_range[0] > 10 and low_res_maxima:
        image_resize_factor = 10.0 / float(size_range[0])
        if automatic_suppression:
            maxima_suppression_size = 7
        else:
            maxima_suppression_size = (
                maxima_suppression_size * image_resize_factor + 0.5
            )
        reported_maxima_suppression_size = (
            maxima_suppression_size / image_resize_factor
        )
    else:
        image_resize_factor = 1.0
        if automatic_suppression:
            maxima_suppression_size = size_range[0] / 1.5
        else:
            maxima_suppression_size = maxima_suppression_size
        reported_maxima_suppression_size = maxima_suppression_size
    maxima_mask = centrosome.cpmorphology.strel_disk(
        max(1, maxima_suppression_size - 0.5)
    )
    distance_transformed_image = None

    if unclump_method == "intensity":
        # Remove dim maxima
        maxima_image = get_maxima(
            blurred_image, labeled_image, maxima_mask, image_resize_factor
        )
    elif unclump_method == "shape":
        if fill_holes == "never":
            # For shape, even if the user doesn't want to fill holes,
            # a point far away from the edge might be near a hole.
            # So we fill just for this part.
            foreground = (
                centrosome.cpmorphology.fill_labeled_holes(labeled_image) > 0
            )
        else:
            foreground = labeled_image > 0
        distance_transformed_image = scipy.ndimage.distance_transform_edt(
            foreground
        )
        # randomize the distance slightly to get unique maxima
        np.random.seed(0)
        distance_transformed_image += np.random.uniform(
            0, 0.001, distance_transformed_image.shape
        )
        maxima_image = get_maxima(
            distance_transformed_image,
            labeled_image,
            maxima_mask,
            image_resize_factor,
        )
    else:
        raise ValueError(
            "Unsupported local maxima method: %s" % unclump_method
        )

    # Create the image for watershed
    if watershed_method == "intensity":
        # use the reverse of the image to get valleys at peaks
        watershed_image = 1 - image
    elif watershed_method == "shape":
        if distance_transformed_image is None:
            distance_transformed_image = scipy.ndimage.distance_transform_edt(
                labeled_image > 0
            )
        watershed_image = -distance_transformed_image
        watershed_image = watershed_image - np.min(watershed_image)
    elif watershed_method == "propagate":
        # No image used
        pass
    else:
        raise NotImplementedError(
            "Watershed method %s is not implemented" % watershed_method
        )
    #
    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    labeled_maxima, object_count = scipy.ndimage.label(
        maxima_image, np.ones((3, 3), bool)
    )
    if watershed_method == "propagate":
        watershed_boundaries, distance = centrosome.propagate.propagate(
            np.zeros(labeled_maxima.shape),
            labeled_maxima,
            labeled_image != 0,
            1.0,
        )
    else:
        markers_dtype = (
            np.int16
            if object_count < np.iinfo(np.int16).max
            else np.int32
        )
        markers = np.zeros(watershed_image.shape, markers_dtype)
        markers[labeled_maxima > 0] = -labeled_maxima[
            labeled_maxima > 0
        ]

        #
        # Some labels have only one maker in them, some have multiple and
        # will be split up.
        #

        watershed_boundaries = skimage.segmentation.watershed(
            connectivity=np.ones((3, 3), bool),
            image=watershed_image,
            markers=markers,
            mask=labeled_image != 0,
        )

        watershed_boundaries = -watershed_boundaries

    return watershed_boundaries, object_count, reported_maxima_suppression_size


def calc_smoothing_filter_size(automatic_smoothing, size_range, smoothing_filter_size):
    """Return the size of the smoothing filter, calculating it if in automatic mode"""
    if automatic_smoothing:
        return 2.35 * size_range[0] / 3.5
    else:
        return smoothing_filter_size


def smooth_image(image, mask, automatic_smoothing, size_range, smoothing_filter_size):
    """Apply the smoothing filter to the image"""

    filter_size = calc_smoothing_filter_size(
        automatic_smoothing, size_range, smoothing_filter_size)
    if filter_size == 0:
        return image
    sigma = filter_size / 2.35
    #
    # We not only want to smooth using a Gaussian, but we want to limit
    # the spread of the smoothing to 2 SD, partly to make things happen
    # locally, partly to make things run faster, partly to try to match
    # the Matlab behavior.
    #
    filter_size = max(int(float(filter_size) / 2.0), 1)
    f = (
        1
        / np.sqrt(2.0 * np.pi)
        / sigma
        * np.exp(
            -0.5 * np.arange(-filter_size, filter_size +
                             1) ** 2 / sigma ** 2
        )
    )

    def fgaussian(image):
        output = scipy.ndimage.convolve1d(
            image, f, axis=0, mode="constant")
        return scipy.ndimage.convolve1d(output, f, axis=1, mode="constant")

        #
        # Use the trick where you similarly convolve an array of ones to find
        # out the edge effects, then divide to correct the edge effects
        #
    edge_array = fgaussian(mask.astype(float))
    masked_image = image.copy()
    masked_image[~mask] = 0
    smoothed_image = fgaussian(masked_image)
    masked_image[mask] = smoothed_image[mask] / edge_array[mask]
    return masked_image


def get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
    if image_resize_factor < 1.0:
        shape = np.array(image.shape) * image_resize_factor
        i_j = (
            np.mgrid[0: shape[0], 0: shape[1]].astype(float)
            / image_resize_factor
        )
        resized_image = scipy.ndimage.map_coordinates(image, i_j)
        resized_labels = scipy.ndimage.map_coordinates(
            labeled_image, i_j, order=0
        ).astype(labeled_image.dtype)

    else:
        resized_image = image
        resized_labels = labeled_image
    #
    # find local maxima
    #
    if maxima_mask is not None:
        binary_maxima_image = centrosome.cpmorphology.is_local_maximum(
            resized_image, resized_labels, maxima_mask
        )
        binary_maxima_image[resized_image <= 0] = 0
    else:
        binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
    if image_resize_factor < 1.0:
        inverse_resize_factor = float(image.shape[0]) / float(
            binary_maxima_image.shape[0]
        )
        i_j = (
            np.mgrid[0: image.shape[0], 0: image.shape[1]].astype(float)
            / inverse_resize_factor
        )
        binary_maxima_image = (
            scipy.ndimage.map_coordinates(
                binary_maxima_image.astype(float), i_j)
            > 0.5
        )
        assert binary_maxima_image.shape[0] == image.shape[0]
        assert binary_maxima_image.shape[1] == image.shape[1]

    # Erode blobs of touching maxima to a single point

    shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
    return shrunk_image
