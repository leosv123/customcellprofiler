import numpy as np
import scipy


def filter_on_border(image, exclude_border_objects, labeled_image):
    """Filter out objects touching the border
    In addition, if the image has a mask, filter out objects
    touching the border of the mask.
    """
    if exclude_border_objects:
        border_labels = list(labeled_image[0, :])
        border_labels.extend(labeled_image[:, 0])
        border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
        border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
        border_labels = np.array(border_labels)
        #
        # the following histogram has a value > 0 for any object
        # with a border pixel
        #
        histogram = scipy.sparse.coo_matrix(
            (
                np.ones(border_labels.shape),
                (border_labels, np.zeros(border_labels.shape)),
            ),
            shape=(np.max(labeled_image) + 1, 1),
        ).todense()
        histogram = np.array(histogram).flatten()
        if any(histogram[1:] > 0):
            histogram_image = histogram[labeled_image]
            labeled_image[histogram_image > 0] = 0
        elif image.has_mask:
            # The assumption here is that, if nothing touches the border,
            # the mask is a large, elliptical mask that tells you where the
            # well is. That's the way the old Matlab code works and it's duplicated here
            #
            # The operation below gets the mask pixels that are on the border of the mask
            # The erosion turns all pixels touching an edge to zero. The not of this
            # is the border + formerly masked-out pixels.
            mask_border = np.logical_not(
                scipy.ndimage.binary_erosion(image.mask)
            )
            mask_border = np.logical_and(mask_border, image.mask)
            border_labels = labeled_image[mask_border]
            border_labels = border_labels.flatten()
            histogram = scipy.sparse.coo_matrix(
                (
                    np.ones(border_labels.shape),
                    (border_labels, np.zeros(border_labels.shape)),
                ),
                shape=(np.max(labeled_image) + 1, 1),
            ).todense()
            histogram = np.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
    return labeled_image


def filter_on_size(labeled_image, exclude_size, object_count, size_range):
    """ Filter the labeled image based on the size range
    labeled_image - pixel image labels
    object_count - # of objects in the labeled image
    returns the labeled image, and the labeled image with the
    small objects removed
    """
    if exclude_size and object_count > 0:
        areas = scipy.ndimage.measurements.sum(
            np.ones(labeled_image.shape),
            labeled_image,
            np.array(list(range(0, object_count + 1)), dtype=np.int32),
        )
        areas = np.array(areas, dtype=int)
        min_allowed_area = (
            np.pi * (size_range[0] * size_range[0]) / 4
        )
        max_allowed_area = (
            np.pi * (size_range[1] * size_range[1]) / 4
        )
        # area_image has the area of the object at every pixel within the object
        area_image = areas[labeled_image]
        labeled_image[area_image < min_allowed_area] = 0
        small_removed_labels = labeled_image.copy()
        labeled_image[area_image > max_allowed_area] = 0
    else:
        small_removed_labels = labeled_image.copy()
    return labeled_image, small_removed_labels
