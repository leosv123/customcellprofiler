import numpy
import pandas

from Imageclass import Image
import skimage.exposure


class RescaleIntensity:
    def __init__(self):
        pass

    def rescale(self, image, in_range, out_range=(0.0, 1.0)):
        data = 1.0 * image["pixel_data"]

        rescaled = skimage.exposure.rescale_intensity(
            data, in_range=in_range, out_range=out_range
        )

        return rescaled

    def stretch(self, input_image):
        data = input_image["pixel_data"]
        mask = input_image["mask"]

        if input_image["multichannel"]:
            print("Hello", input_image["multichannel"])
            splitaxis = data.ndim - 1
            singlechannels = numpy.split(data, data.shape[-1], splitaxis)
            newchannels = []
            for channel in singlechannels:
                channel = numpy.squeeze(channel, axis=splitaxis)
                if (masked_channel := channel[mask]).size == 0:
                    in_range = (0, 1)
                else:
                    in_range = (min(masked_channel), max(masked_channel))

                channelholder = Image(channel, convert=False)

                rescaled = self.rescale(channelholder, in_range)
                newchannels.append(rescaled)
            full_rescaled = numpy.stack(newchannels, axis=-1)
            return full_rescaled
        if (masked_data := data[mask]).size == 0:
            in_range = (0, 1)
        else:
            in_range = (min(masked_data), max(masked_data))
        return self.rescale(input_image, in_range)
