import argparse
import tensorflow as tf
import os

from PIL import Image

import numpy as np
import pandas as pd
import copy

import matplotlib
import matplotlib.pyplot as plt
#from libtiff import TIFF
from skimage import io
import cv2
screen = 'Plate13'

data = pd.read_csv('cellprofiler_output.csv')
print(data.shape[0])
data = data[data['yeast_AreaShape_Area'] > 999]
#data = data[data['Image_FileName_GFP2']=='014009-1-001001004.tif']
n_cell = data.shape[0]


print(n_cell)


try:
    os.mkdir('./' + screen + '/cell_print_xy')
except OSError:
    print("Creation of the directory failed")
else:
    print("Successfully created the directory ")


sql_col_names = ['ImageNumber', 'Image_FileName_GFP1', 'Image_FileName_GFP2',

                 'yeast_AreaShape_Center_X', 'yeast_AreaShape_Center_Y', 'yeast_AreaShape_Area',
                 'yeast_Intensity_IntegratedIntensity_ResizeGFP2', 'yeast_Intensity_MeanIntensity_ResizeGFP2',
                 'yeast_Intensity_StdIntensity_ResizeGFP2', 'yeast_Intensity_MinIntensity_ResizeGFP2',
                 'yeast_Intensity_MaxIntensity_ResizeGFP2', 'yeast_Intensity_IntegratedIntensityEdge_ResizeGFP2']


basePath = '/Users/lingrajsvannur/Downloads/Resized_1_5_linear/'

#GFP_images = np.unique(self.sql_data['Image_FileName_GFP2'])
# GFP_images.sort()
#self.wells = np.unique([seq[0:8] for seq in GFP_images])

#self.strains = np.unique([seq[0:6] for seq in GFP_images])
# print(self.strains)
cropSize = 60


imSize = 64
numClasses = 19
numChan = 1


croppedCells = np.zeros((n_cell, imSize ** 2 * 2))
coordUsed = np.zeros((n_cell, 2))

ind = 0

#selected_wells = []
wellNames = []

GFP_images = np.unique(data['Image_FileName_GFP2'])
GFP_images.sort()
wells = np.unique([seq[0:8] for seq in GFP_images])
for well in wells:
    print(well)
    # if well[0:6] != prewell[0:6]:
    # break
    #G_array = G_arrays[int(frame/2)]
    #R_array = R_arrays[int(frame/2)]
    G_array = io.imread(basePath+well+'-001001004.tif')
    #R_array = io.imread(basePath+'Resized_1_5_linear/'+well+'-001001003.tif')
    B_array = io.imread(basePath+well+'-001001001.tif')
    #GFPImg_L=GFPImg.resize((2160, 2160),Image.NEAREST)
    # GFPImg.save(self.basePath+'tif_file/'+well+'-001001002thumbnail.tif')
    #GFPImg_L = Image.open(self.basePath+'tif_file/'+well+'-001001002thumbnail.tif')
    #G_array = GFPImg.read_image()
    print(G_array.shape)
    curCoordinates = data[data['Image_FileName_GFP2'] == well + '-001001004.tif'][
        ['yeast_AreaShape_Center_X',
         'yeast_AreaShape_Center_Y']]

    #RFPImg = Image.open(self.basePath+'resized_1_5/'+well+'-001001003processed_boundary.tiff')
    #RFPImg_L=RFPImg.resize((2160, 2160),Image.NEAREST)
    #R_array = np.zeros((G_array.shape))

    coord = 0

    while coord < len(curCoordinates):
        cur_y, cur_x = curCoordinates.values[coord]

        if (cur_x - imSize / 2 > 0 and cur_x + imSize / 2 < G_array.shape[0] and
           cur_y - imSize / 2 > 0 and cur_y + imSize / 2 < G_array.shape[1]):
            croppedCells[ind, : imSize ** 2] = (
                G_array[int(np.floor(cur_x - imSize / 2)):int(np.floor(cur_x + imSize / 2)),
                        int(np.floor(cur_y - imSize / 2)):int(np.floor(cur_y + imSize / 2))]).ravel()

            croppedCells[ind, imSize ** 2:] = (
                B_array[int(np.floor(cur_x - imSize / 2)):int(np.floor(cur_x + imSize / 2)),
                        int(np.floor(cur_y - imSize / 2)):int(np.floor(cur_y + imSize / 2))]).ravel()

            coordUsed[ind, :] = [cur_y, cur_x]

            coord += 1

            wellNames.append(well)

            cell_arrayB = B_array[int(np.floor(cur_x - imSize / 2)):int(np.floor(
                cur_x + imSize / 2)), int(np.floor(cur_y - imSize / 2)):int(np.floor(cur_y + imSize / 2))]
            cell_arrayG = G_array[int(np.floor(cur_x - imSize / 2)):int(np.floor(
                cur_x + imSize / 2)), int(np.floor(cur_y - imSize / 2)):int(np.floor(cur_y + imSize / 2))]
            #rgbArray = np.zeros((64,64,3), 'uint8')
            #rgbArray[..., 0]=cell_arrayR
            #rgbArray[..., 1]=cell_arrayG
            #rgbArray[..., 2]=np.zeros(cell_arrayR.shape)
            cell_GFP = Image.fromarray(cell_arrayG)
            cell_nucli = Image.fromarray(cell_arrayB)

            cell_GFP.save(basePath+'cell_print_xy/'+well+'_' +
                          str(cur_y)+'_'+str(cur_x)+'_GFP.tif')
            cell_nucli.save(basePath+'cell_print_xy/'+well +
                            '_'+str(cur_y)+'_'+str(cur_x)+'_nucli.tif')
            ind += 1

        else:
            coord += 1


curCroppedCells = croppedCells[:ind]
print(n_cell)
print(croppedCells.shape)
#intensityUsed = intensityUsed[:ind]
print(coordUsed.shape)
