import cellprofiler_core.module.image_segmentation
import cellprofiler_core.object
from threshold import get_threshold_manual, apply_threshold, add_fg_bg_measurements
from separate_neighboring_objects import separate_neighboring_objects, calc_smoothing_filter_size
from filter_on_border_size import filter_on_border, filter_on_size
import numpy as np
import scipy
import centrosome.smooth
import centrosome.threshold
import scipy.interpolate
import scipy.ndimage
import skimage.filters
import skimage.filters.rank
import skimage.morphology
from omegaconf import OmegaConf
from argparse import ArgumentParser
import math
import warnings
warnings.filterwarnings('ignore')


class IdentifyPrimaryObject:
    def __init__(self, config_data) -> None:

        self.input_path = config_data.config.image_path
        self.threshold_scope = config_data.config.threshold_scope
        self.automatic = False
        if self.threshold_scope == 'Global':
            self.name_objects = config_data.config.Global["name_objects"]
            self.size_range = config_data.config.Global["size_range"]
            self.exclude_size = config_data.config.Global["exclude_size"]
            self.exclude_border_objects = config_data.config.Global["exclude_border_objects"]
            self.global_operation = config_data.config.Global["global_operation"]
            self.manual_threshold = config_data.config.Global["manual_threshold"]
            self.threshold_smoothing_scale = config_data.config.Global["threshold_smoothing_scale"]
            self.unclump_method = config_data.config.Global["unclump_method"]
            self.watershed_method = config_data.config.Global["watershed_method"]
            self.automatic_smoothing = config_data.config.Global["automatic_smoothing"]
            self.automatic_suppression = config_data.config.Global["automatic_suppression"]
            self.low_res_maxima = config_data.config.Global["low_res_maxima"]
            self.fill_holes = config_data.config.Global["fill_holes"]
            self.want_plot_maxima = config_data.config.Global["want_plot_maxima"]
            self.limit_choice = config_data.config.Global["limit_choice"]

    def read_image_fn(self):
        image = skimage.io.imread(self.input_path)
        mask = image != 1
        self.input_image = {'pixel_data': image,
                            "mask": mask, "multichannel": 0}
        print("Input image dictionary:", self.input_image)

    def threshold_fn(self):
        if self.global_operation == "Manual":
            final_threshold, orig_threshold, guide_threshold = get_threshold_manual(
                self.input_image, self.manual_threshold, self.automatic)

        binary_image, sigma = apply_threshold(
            self.input_image, self.threshold_smoothing_scale, final_threshold, self.automatic
        )

        add_fg_bg_measurements_dict = add_fg_bg_measurements(
            self.name_objects, self.input_image, binary_image)

        return binary_image, np.mean(np.atleast_1d(final_threshold)), sigma, add_fg_bg_measurements_dict

    def fill_holes_fn(self, binary_image, global_threshold):
        finalstats = {}

        def size_fn(size, is_foreground):
            return size < self.size_range[1] * self.size_range[0]

        if self.fill_holes == "both":
            binary_image = centrosome.cpmorphology.fill_labeled_holes(
                binary_image, size_fn=size_fn
            )
        labeled_image, object_count = scipy.ndimage.label(
            binary_image, np.ones((3, 3), bool)
        )

        (labeled_image, object_count, maxima_suppression_size) = separate_neighboring_objects(self.input_image, self.size_range, self.automatic_smoothing, self.low_res_maxima, self.automatic_suppression, labeled_image,
                                                                                              self.unclump_method, self.fill_holes, object_count, self.watershed_method, smoothing_filter_size=None, maxima_suppression_size=None)

        print("Object COUNT", object_count)

        unedited_labels = labeled_image.copy()

        # Filter out objects touching the border or mask
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = filter_on_border(
            self.input_image, self.exclude_border_objects, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0

        # Filter out small and large objects
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = filter_on_size(
            labeled_image, self.exclude_size, object_count, self.size_range
        )
        size_excluded_labeled_image[labeled_image > 0] = 0

        print("Till Now", labeled_image)

        if self.fill_holes != "never":
            labeled_image = centrosome.cpmorphology.fill_labeled_holes(
                labeled_image)

        # Relabel the image
        labeled_image, object_count = centrosome.cpmorphology.relabel(
            labeled_image)

        if self.limit_choice == "erase":
            if object_count > self.maximum_object_count.value:
                labeled_image = np.zeros(labeled_image.shape, int)
                border_excluded_labeled_image = np.zeros(
                    labeled_image.shape, int)
                size_excluded_labeled_image = np.zeros(
                    labeled_image.shape, int)
                object_count = 0

        # Make an outline image
        outline_image = centrosome.outline.outline(labeled_image)
        outline_size_excluded_image = centrosome.outline.outline(
            size_excluded_labeled_image
        )
        outline_border_excluded_image = centrosome.outline.outline(
            border_excluded_labeled_image
        )

        finalstats['# of accepted objects'] = object_count
        if object_count > 0:
            print("cool")
            areas = scipy.ndimage.sum(
                np.ones(labeled_image.shape),
                labeled_image,
                np.arange(1, object_count + 1),
            )
            areas.sort()
            low_diameter = (
                math.sqrt(float(areas[object_count // 10]) / np.pi) * 2
            )
            median_diameter = (
                math.sqrt(float(areas[object_count // 2]) / np.pi) * 2
            )
            high_diameter = (
                math.sqrt(float(areas[object_count * 9 // 10]) / np.pi) * 2
            )

            finalstats['10th pctile diameter'] = low_diameter
            finalstats['Median diameter'] = median_diameter
            finalstats['90th pctile diameter'] = high_diameter
            object_area = np.sum(areas)
            total_area = np.product(labeled_image.shape[:2])
            finalstats['Area covered by objects'] = float(
                object_area)/float(total_area)
        else:
            finalstats['10th pctile diameter'] = None
            finalstats['Median diameter'] = None
            finalstats['90th pctile diameter'] = None
            object_area = None
            total_area = None
            finalstats['Area covered by objects'] = None

        finalstats['Thresholding filter size'] = global_threshold

        if self.unclump_method != 'none':
            finalstats["Declumping smoothing filter size"] = calc_smoothing_filter_size(
                self.automatic_smoothing, self.size_range, smoothing_filter_size=None)
            finalstats["Maxima suppression size"] = maxima_suppression_size

        else:
            finalstats["Threshold"] = global_threshold

        objname = self.name_objects
        objects = cellprofiler_core.object.Objects()
        objects.segmented = labeled_image
        objects.unedited_segmented = unedited_labels
        objects.small_removed_segmented = small_removed_labels
        objects.parent_image = self.input_image

        return finalstats, labeled_image, outline_image, objects


if __name__ == '__main__':
    # get the image --- Done
    # Threshold the image
    # Fill background holes inside foreground objects
    # size function
    # add measurements ( record them to a csv file)
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        required=True, help="provide the config file")
    args = parser.parse_args()
    # get configuration
    configFile = OmegaConf.load(args.config)
    po = IdentifyPrimaryObject(config_data=configFile)
    po.read_image_fn()
    binary_image, global_threshold, sigma, add_fg_bg_measurements_dict = po.threshold_fn()
    finalstats, labeled_image, outline_image = po.fill_holes_fn(
        binary_image, global_threshold)
    print(finalstats)
