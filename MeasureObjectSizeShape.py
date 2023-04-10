
import centrosome.cpmorphology
import centrosome.zernike
import numpy
import scipy.ndimage
import skimage.measure
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects

AREA_SHAPE = "AreaShape"

"""Calculate Zernike features for N,M where N=0 through ZERNIKE_N"""
ZERNIKE_N = 9

F_AREA = "Area"
F_PERIMETER = "Perimeter"
F_VOLUME = "Volume"
F_SURFACE_AREA = "SurfaceArea"
F_ECCENTRICITY = "Eccentricity"
F_SOLIDITY = "Solidity"
F_CONVEX_AREA = "ConvexArea"
F_EXTENT = "Extent"
F_CENTER_X = "Center_X"
F_CENTER_Y = "Center_Y"
F_CENTER_Z = "Center_Z"
F_BBOX_AREA = "BoundingBoxArea"
F_BBOX_VOLUME = "BoundingBoxVolume"
F_MIN_X = "BoundingBoxMinimum_X"
F_MAX_X = "BoundingBoxMaximum_X"
F_MIN_Y = "BoundingBoxMinimum_Y"
F_MAX_Y = "BoundingBoxMaximum_Y"
F_MIN_Z = "BoundingBoxMinimum_Z"
F_MAX_Z = "BoundingBoxMaximum_Z"
F_EULER_NUMBER = "EulerNumber"
F_FORM_FACTOR = "FormFactor"
F_MAJOR_AXIS_LENGTH = "MajorAxisLength"
F_MINOR_AXIS_LENGTH = "MinorAxisLength"
F_ORIENTATION = "Orientation"
F_COMPACTNESS = "Compactness"
F_INERTIA = "InertiaTensor"
F_MAXIMUM_RADIUS = "MaximumRadius"
F_MEDIAN_RADIUS = "MedianRadius"
F_MEAN_RADIUS = "MeanRadius"
F_MIN_FERET_DIAMETER = "MinFeretDiameter"
F_MAX_FERET_DIAMETER = "MaxFeretDiameter"

F_CENTRAL_MOMENT_0_0 = "CentralMoment_0_0"
F_CENTRAL_MOMENT_0_1 = "CentralMoment_0_1"
F_CENTRAL_MOMENT_0_2 = "CentralMoment_0_2"
F_CENTRAL_MOMENT_0_3 = "CentralMoment_0_3"
F_CENTRAL_MOMENT_1_0 = "CentralMoment_1_0"
F_CENTRAL_MOMENT_1_1 = "CentralMoment_1_1"
F_CENTRAL_MOMENT_1_2 = "CentralMoment_1_2"
F_CENTRAL_MOMENT_1_3 = "CentralMoment_1_3"
F_CENTRAL_MOMENT_2_0 = "CentralMoment_2_0"
F_CENTRAL_MOMENT_2_1 = "CentralMoment_2_1"
F_CENTRAL_MOMENT_2_2 = "CentralMoment_2_2"
F_CENTRAL_MOMENT_2_3 = "CentralMoment_2_3"
F_EQUIVALENT_DIAMETER = "EquivalentDiameter"
F_HU_MOMENT_0 = "HuMoment_0"
F_HU_MOMENT_1 = "HuMoment_1"
F_HU_MOMENT_2 = "HuMoment_2"
F_HU_MOMENT_3 = "HuMoment_3"
F_HU_MOMENT_4 = "HuMoment_4"
F_HU_MOMENT_5 = "HuMoment_5"
F_HU_MOMENT_6 = "HuMoment_6"
F_INERTIA_TENSOR_0_0 = "InertiaTensor_0_0"
F_INERTIA_TENSOR_0_1 = "InertiaTensor_0_1"
F_INERTIA_TENSOR_1_0 = "InertiaTensor_1_0"
F_INERTIA_TENSOR_1_1 = "InertiaTensor_1_1"
F_INERTIA_TENSOR_EIGENVALUES_0 = "InertiaTensorEigenvalues_0"
F_INERTIA_TENSOR_EIGENVALUES_1 = "InertiaTensorEigenvalues_1"
F_NORMALIZED_MOMENT_0_0 = "NormalizedMoment_0_0"
F_NORMALIZED_MOMENT_0_1 = "NormalizedMoment_0_1"
F_NORMALIZED_MOMENT_0_2 = "NormalizedMoment_0_2"
F_NORMALIZED_MOMENT_0_3 = "NormalizedMoment_0_3"
F_NORMALIZED_MOMENT_1_0 = "NormalizedMoment_1_0"
F_NORMALIZED_MOMENT_1_1 = "NormalizedMoment_1_1"
F_NORMALIZED_MOMENT_1_2 = "NormalizedMoment_1_2"
F_NORMALIZED_MOMENT_1_3 = "NormalizedMoment_1_3"
F_NORMALIZED_MOMENT_2_0 = "NormalizedMoment_2_0"
F_NORMALIZED_MOMENT_2_1 = "NormalizedMoment_2_1"
F_NORMALIZED_MOMENT_2_2 = "NormalizedMoment_2_2"
F_NORMALIZED_MOMENT_2_3 = "NormalizedMoment_2_3"
F_NORMALIZED_MOMENT_3_0 = "NormalizedMoment_3_0"
F_NORMALIZED_MOMENT_3_1 = "NormalizedMoment_3_1"
F_NORMALIZED_MOMENT_3_2 = "NormalizedMoment_3_2"
F_NORMALIZED_MOMENT_3_3 = "NormalizedMoment_3_3"
F_SPATIAL_MOMENT_0_0 = "SpatialMoment_0_0"
F_SPATIAL_MOMENT_0_1 = "SpatialMoment_0_1"
F_SPATIAL_MOMENT_0_2 = "SpatialMoment_0_2"
F_SPATIAL_MOMENT_0_3 = "SpatialMoment_0_3"
F_SPATIAL_MOMENT_1_0 = "SpatialMoment_1_0"
F_SPATIAL_MOMENT_1_1 = "SpatialMoment_1_1"
F_SPATIAL_MOMENT_1_2 = "SpatialMoment_1_2"
F_SPATIAL_MOMENT_1_3 = "SpatialMoment_1_3"
F_SPATIAL_MOMENT_2_0 = "SpatialMoment_2_0"
F_SPATIAL_MOMENT_2_1 = "SpatialMoment_2_1"
F_SPATIAL_MOMENT_2_2 = "SpatialMoment_2_2"
F_SPATIAL_MOMENT_2_3 = "SpatialMoment_2_3"

"""The non-Zernike features"""
F_STD_2D = [
    F_AREA,
    F_PERIMETER,
    F_MAXIMUM_RADIUS,
    F_MEAN_RADIUS,
    F_MEDIAN_RADIUS,
    F_MIN_FERET_DIAMETER,
    F_MAX_FERET_DIAMETER,
    F_ORIENTATION,
    F_ECCENTRICITY,
    F_FORM_FACTOR,
    F_SOLIDITY,
    F_CONVEX_AREA,
    F_COMPACTNESS,
    F_BBOX_AREA,
]
F_STD_3D = [
    F_VOLUME,
    F_SURFACE_AREA,
    F_CENTER_Z,
    F_BBOX_VOLUME,
    F_MIN_Z,
    F_MAX_Z,
]
F_ADV_2D = [
    F_SPATIAL_MOMENT_0_0,
    F_SPATIAL_MOMENT_0_1,
    F_SPATIAL_MOMENT_0_2,
    F_SPATIAL_MOMENT_0_3,
    F_SPATIAL_MOMENT_1_0,
    F_SPATIAL_MOMENT_1_1,
    F_SPATIAL_MOMENT_1_2,
    F_SPATIAL_MOMENT_1_3,
    F_SPATIAL_MOMENT_2_0,
    F_SPATIAL_MOMENT_2_1,
    F_SPATIAL_MOMENT_2_2,
    F_SPATIAL_MOMENT_2_3,
    F_CENTRAL_MOMENT_0_0,
    F_CENTRAL_MOMENT_0_1,
    F_CENTRAL_MOMENT_0_2,
    F_CENTRAL_MOMENT_0_3,
    F_CENTRAL_MOMENT_1_0,
    F_CENTRAL_MOMENT_1_1,
    F_CENTRAL_MOMENT_1_2,
    F_CENTRAL_MOMENT_1_3,
    F_CENTRAL_MOMENT_2_0,
    F_CENTRAL_MOMENT_2_1,
    F_CENTRAL_MOMENT_2_2,
    F_CENTRAL_MOMENT_2_3,
    F_NORMALIZED_MOMENT_0_0,
    F_NORMALIZED_MOMENT_0_1,
    F_NORMALIZED_MOMENT_0_2,
    F_NORMALIZED_MOMENT_0_3,
    F_NORMALIZED_MOMENT_1_0,
    F_NORMALIZED_MOMENT_1_1,
    F_NORMALIZED_MOMENT_1_2,
    F_NORMALIZED_MOMENT_1_3,
    F_NORMALIZED_MOMENT_2_0,
    F_NORMALIZED_MOMENT_2_1,
    F_NORMALIZED_MOMENT_2_2,
    F_NORMALIZED_MOMENT_2_3,
    F_NORMALIZED_MOMENT_3_0,
    F_NORMALIZED_MOMENT_3_1,
    F_NORMALIZED_MOMENT_3_2,
    F_NORMALIZED_MOMENT_3_3,
    F_HU_MOMENT_0,
    F_HU_MOMENT_1,
    F_HU_MOMENT_2,
    F_HU_MOMENT_3,
    F_HU_MOMENT_4,
    F_HU_MOMENT_5,
    F_HU_MOMENT_6,
    F_INERTIA_TENSOR_0_0,
    F_INERTIA_TENSOR_0_1,
    F_INERTIA_TENSOR_1_0,
    F_INERTIA_TENSOR_1_1,
    F_INERTIA_TENSOR_EIGENVALUES_0,
    F_INERTIA_TENSOR_EIGENVALUES_1,
]
F_ADV_3D = [F_SOLIDITY]
F_STANDARD = [
    F_EXTENT,
    F_EULER_NUMBER,
    F_EQUIVALENT_DIAMETER,
    F_MAJOR_AXIS_LENGTH,
    F_MINOR_AXIS_LENGTH,
    F_CENTER_X,
    F_CENTER_Y,
    F_MIN_X,
    F_MIN_Y,
    F_MAX_X,
    F_MAX_Y,
]


class MeasureObjects:
    def __init__(self, object_name, calculate_zernikes=False):
        self.calculate_zernikes = calculate_zernikes
        self.object_name = object_name

    def get_zernike_numbers(self):
        """The Zernike numbers measured by this module"""
        if self.calculate_zernikes:
            return centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)
        else:
            return []

    def record_measurement(self, object_name, feature_name, result):
        """Record the result of a measurement in the workspace's measurements"""
        t = {}
        data = centrosome.cpmorphology.fixup_scipy_ndimage_result(result)
        t[object_name, "%s_%s" % (AREA_SHAPE, feature_name)] = data

        if numpy.any(numpy.isfinite(data)) > 0:
            data = data[numpy.isfinite(data)]
            t['mean'] = numpy.mean(data)
            t['median'] = numpy.median(data)
            t['std'] = numpy.std(data)
            t['object_name'] = object_name
            t['feature_name'] = feature_name
        return t

    def analyze_objects(self, objects, desired_properties):
        """Computing the measurements for a single map of objects"""
        labels = objects.segmented
        nobjects = len(objects.indices)
        if len(objects.shape) == 2:
            props = skimage.measure.regionprops_table(
                labels, properties=desired_properties
            )

            formfactor = 4.0 * numpy.pi * \
                props["area"] / props["perimeter"] ** 2
            denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
            compactness = props["perimeter"] ** 2 / denom

            max_radius = numpy.zeros(nobjects)
            median_radius = numpy.zeros(nobjects)
            mean_radius = numpy.zeros(nobjects)
            min_feret_diameter = numpy.zeros(nobjects)
            max_feret_diameter = numpy.zeros(nobjects)
            zernike_numbers = self.get_zernike_numbers()

            zf = {}
            for n, m in zernike_numbers:
                zf[(n, m)] = numpy.zeros(nobjects)

            for index, mini_image in enumerate(props["image"]):
                # Pad image to assist distance tranform
                mini_image = numpy.pad(mini_image, 1)
                distances = scipy.ndimage.distance_transform_edt(mini_image)
                max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum(distances, mini_image)
                )
                mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.mean(distances, mini_image)
                )
                median_radius[index] = centrosome.cpmorphology.median_of_labels(
                    distances, mini_image.astype("int"), [1]
                )
            #
            # Zernike features
            #
            if self.calculate_zernikes:
                zf_l = centrosome.zernike.zernike(
                    zernike_numbers, labels, objects.indices
                )
                for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                    zf[(n, m)] = z

            if nobjects > 0:
                chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(
                    objects.ijv, objects.indices
                )
                #
                # Feret diameter
                #
                (
                    min_feret_diameter,
                    max_feret_diameter,
                ) = centrosome.cpmorphology.feret_diameter(
                    chulls, chull_counts, objects.indices
                )

            features_to_record = {
                F_AREA: props["area"],
                F_PERIMETER: props["perimeter"],
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_ECCENTRICITY: props["eccentricity"],
                F_ORIENTATION: props["orientation"] * (180 / numpy.pi),
                F_CENTER_X: props["centroid-1"],
                F_CENTER_Y: props["centroid-0"],
                F_BBOX_AREA: props["bbox_area"],
                F_MIN_X: props["bbox-1"],
                F_MAX_X: props["bbox-3"],
                F_MIN_Y: props["bbox-0"],
                F_MAX_Y: props["bbox-2"],
                F_FORM_FACTOR: formfactor,
                F_EXTENT: props["extent"],
                F_SOLIDITY: props["solidity"],
                F_COMPACTNESS: compactness,
                F_EULER_NUMBER: props["euler_number"],
                F_MAXIMUM_RADIUS: max_radius,
                F_MEAN_RADIUS: mean_radius,
                F_MEDIAN_RADIUS: median_radius,
                F_CONVEX_AREA: props["convex_area"],
                F_MIN_FERET_DIAMETER: min_feret_diameter,
                F_MAX_FERET_DIAMETER: max_feret_diameter,
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
            }

            if self.calculate_zernikes:
                features_to_record.update(
                    {
                        self.get_zernike_name((n, m)): zf[(n, m)]
                        for n, m in zernike_numbers
                    }
                )

        else:

            props = skimage.measure.regionprops_table(
                labels, properties=desired_properties
            )

            # SurfaceArea
            surface_areas = numpy.zeros(len(props["label"]))
            for index, label in enumerate(props["label"]):
                # this seems less elegant than you might wish, given that regionprops returns a slice,
                # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
                volume = labels[max(props['bbox-0'][index]-1, 0):min(props['bbox-3'][index]+1, labels.shape[0]),
                                max(props['bbox-1'][index]-1, 0):min(props['bbox-4'][index]+1, labels.shape[1]),
                                max(props['bbox-2'][index]-1, 0):min(props['bbox-5'][index]+1, labels.shape[2])]
                volume = volume == label
                verts, faces, _normals, _values = skimage.measure.marching_cubes(
                    volume,
                    method="lewiner",
                    spacing=objects.parent_image.spacing
                    if objects.has_parent_image
                    else (1.0,) * labels.ndim,
                    level=0,
                )
                surface_areas[index] = skimage.measure.mesh_surface_area(
                    verts, faces)

            features_to_record = {
                F_VOLUME: props["area"],
                F_SURFACE_AREA: surface_areas,
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_CENTER_X: props["centroid-2"],
                F_CENTER_Y: props["centroid-1"],
                F_CENTER_Z: props["centroid-0"],
                F_BBOX_VOLUME: props["bbox_area"],
                F_MIN_X: props["bbox-2"],
                F_MAX_X: props["bbox-5"],
                F_MIN_Y: props["bbox-1"],
                F_MAX_Y: props["bbox-4"],
                F_MIN_Z: props["bbox-0"],
                F_MAX_Z: props["bbox-3"],
                F_EXTENT: props["extent"],
                F_EULER_NUMBER: props["euler_number"],
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
            }

        fullresult = []
        for f, m in features_to_record.items():
            temp = self.record_measurement(self.object_name, f, m)
            fullresult.append(temp)

        return features_to_record, fullresult
