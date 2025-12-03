import numpy as np
from skimage import io
import os
import skimage
import tqdm

def RGB_ChannelCheck(raw_img):

    RGB_channel = None
    
    if raw_img.ndim==2:
        pass
    elif raw_img.ndim==3:
        RGB_channels = np.flatnonzero(raw_img.reshape(-1, raw_img.shape[-1]).sum(axis=0))
        if len(RGB_channels)==1:
          RGB_channel = RGB_channels[0]
        else:
          raise Exception('Images for cropping should be only 8-bit grayscale or 8-bit R/G/B.')
    else:
        raise Exception('Images for cropping should be only 8-bit grayscale or 8-bit R/G/B.')

    return RGB_channel

def cropping_cells(mask_img, raw_img, save_path, mask_name, save=True):
    x_min_img = []
    x_max_img = []
    y_min_img = []
    y_max_img = []

    crops = []
    labels = np.empty(0, dtype=int)
    coordinates = np.empty((0, 4), dtype=int)

    RGB_channel = RGB_ChannelCheck(raw_img)

    for i in np.unique(mask_img)[1:]:
        x_min_img.append(np.where(mask_img==i)[1].min())
        x_max_img.append(np.where(mask_img==i)[1].max())
        y_min_img.append(np.where(mask_img==i)[0].min())
        y_max_img.append(np.where(mask_img==i)[0].max())

    j = 0
    print('<======== Started cropping ========>')

    for i in np.unique(mask_img)[1:]:

        mask_img_bw = np.copy(mask_img)
        raw_img_copy = np.copy(raw_img)

        mask_img_bw[mask_img_bw != i] = 0
        mask_img_bw[mask_img_bw == i] = 1

        if RGB_channel == None:
            raw_img_copy = np.multiply(raw_img_copy, mask_img_bw).astype(np.uint8)
        else:
            raw_img_copy[:,:,RGB_channel] = np.multiply(raw_img_copy[:,:,RGB_channel], mask_img_bw)

        cropped_img = raw_img_copy[y_min_img[j]:y_max_img[j], x_min_img[j]:x_max_img[j]]

        labels = np.append(labels, i)
        coordinates = np.append(coordinates, [[x_min_img[j], x_max_img[j], y_min_img[j], y_max_img[j]]], axis = 0)

        if save:
            filename = f'nucleus_{mask_name[:-4]}_{i:02d}.tif'
            io.imsave(os.path.join(save_path, filename), cropped_img)
            #io.imsave(os.path.join(save_path, 'nucleus_' + mask_name[:-4] + '_' + '%2d'% i + '.tif'), cropped_img)

        j = j+1
        crops.append(cropped_img)

        del raw_img_copy, mask_img_bw
    number_segmented = j
    #print(number_segmented, labels, coordinates)
    return crops, number_segmented, labels, coordinates