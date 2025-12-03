import numpy as np
import scipy
from scipy import ndimage
from mahotas.features import tas, pftas, zernike_moments
from mahotas.features.texture import haralick

import cv2
from PIL import Image, ImageEnhance
import skimage
from skimage.feature import local_binary_pattern
from skimage.measure import label
import os
import time
from skimage import exposure, io
import bandpass_segmentation as bs
from tqdm import tqdm

class Preprocess:
    def __init__(self, path_for_images, max_size=None):
        
        assert isinstance(path_for_images, list), 'Please provide a list of paths for images;'\
                                                   'if there is only one path, put it into list.'
        self.path_for_images = path_for_images
        self.max_size = max_size
        
    def class_name(self):
        return self.__class__.__name__
    
    def calc_mean_var(self):
        
#         assert isinstance(self.path_for_images, list), 'Please provide a list of paths for images;'\
#                                                    'if there is only one path, put it into list.'
        num_folders = len(self.path_for_images)
        total_img_size = 0

        for f in range(num_folders):
            for files in os.listdir(self.path_for_images[f]):
                width, height = Image.open(os.path.join(self.path_for_images[f], files)).size
                total_img_size += (width * height)

        allImgs = np.empty(total_img_size, dtype=np.uint8)
        pos = 0
        
        for f in range(num_folders):
            for files in os.listdir(self.path_for_images[f]):
                img = Image.open(os.path.join(self.path_for_images[f], files))
                img = np.array(img)
                img = Image2DConvert(img)
                img = (img - img.min()) / (img - img.min()).max()
                img *= 255
                img = img.astype(np.uint8)
                img = img.ravel()
                size = img.size
                allImgs[pos:pos + size] = img
                pos += size
        
        return allImgs.mean(), allImgs.std()
    
    def mean_var_norm(self, img, mean, std):
        img = (img - mean) / std
        return img      


    def calc_padding(self):
        
#         assert isinstance(self.path_for_images, list), 'Please provide a list of paths for images;'\
#                                                    'if there is only one path, put it into list.'
        
        num_folders = len(self.path_for_images)
        
        max_shape0 = 0
        max_shape1 = 0
        
        for f in range(num_folders):
            for files in os.listdir(self.path_for_images[f]):
                img = Image.open(os.path.join(self.path_for_images[f], files))
                img = np.array(img)

                if img.shape[0] > max_shape0:
                    max_shape0 = img.shape[0]
                if img.shape[1] > max_shape1:
                    max_shape1 = img.shape[1]
                    
        return max_shape0, max_shape1
    
    def padding(self, img, max_shape0, max_shape1):
        
      
        if self.max_size is None:
            self.max_size = max(max_shape0, max_shape1) + 10
        
        
        pad_size = ( ((self.max_size - img.shape[0])//2, (self.max_size - img.shape[0])//2), ((self.max_size - img.shape[1])//2, (self.max_size - img.shape[1])//2) )
        
        img_padded = np.pad(img, pad_size, 'constant')
        
        if img_padded.shape[0] < self.max_size:
            img_padded = np.pad(img_padded, ((self.max_size - img_padded.shape[0], 0), (0,0)), 'constant')
        if img_padded.shape[1] < self.max_size:
            img_padded = np.pad(img_padded, ((0, 0), (0,(self.max_size - img_padded.shape[1]))), 'constant')

        return img_padded
                               
    
class HaralickFeatures:
    
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, image):
     # calculate haralick texture features for 4 types of adjacency
        textures = haralick(image)

       # take the mean and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
    
    
class TAS:
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, img, thresh=None, margin=None):  
        
        if (thresh and margin) is None:
            tas_feat = pftas(img)
        else:
            assert isinstance(thresh(int, float)), 'thresh must be int of float type'

            tas_feat = tas(img, thresh, margin)
        return tas_feat
        

class ZernikeMoments:
    def __init__(self, radius):
        self.radius = radius
        
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, image):
        radius = self.radius
        if radius==None:
            radius = max(image.shape[0], image.shape[1])//2
        if image.max() == 1:
            binary = 1 * (image > 0)
        if image.max() == 255:
            binary = 255 * (image > 0)
        else:
            binary = int(image.max()) * (image > 0)
        return zernike_moments(binary, radius)

class CenterMass:
        
    def class_name(self):
        return self.__class__.__name__
      
    def __call__(self, img):  
        return np.array(ndimage.measurements.center_of_mass(img))
    
class ChromatinFeatures:
    def __init__(self, coeff=1, thresh=0.05):
        self.coeff = coeff
        self.thresh = thresh
        
    def class_name(self):
        return self.__class__.__name__
    
    def __call__(self, img):

        img_adapteq = img

        background = skimage.morphology.area_opening(img_adapteq)
        im_bgadjusted = img_adapteq - background
        filtered = scipy.signal.medfilt(im_bgadjusted)
        segmented_chromatin = bs.bandPassSegm(filtered, 1, 3, self.coeff, self.thresh)
        small_removed  = skimage.morphology.remove_small_holes((segmented_chromatin.max() - segmented_chromatin).astype('bool'), 10)
        small_removed = np.logical_not(small_removed)
        labeled = label(small_removed)
        
        num_chromatine = len(np.unique(labeled)) - 1
        
        min_area = 1000000
        max_area = 0
        
        areas = []
        labeled_copy = labeled.copy()
        for i in np.unique(labeled)[1:]:
            labeled_copy[labeled!=i] = 0
            labeled_copy[labeled==i] = 1
            cur_area = np.sum(labeled_copy>0)
            if cur_area > max_area: max_area = cur_area
            if cur_area < min_area: min_area = cur_area  
            areas.append(cur_area)
        
        if np.shape(areas)[0] > 0:
          mean_area = np.array(areas).mean()
          total_area =  np.sum(np.array(areas))
        else:
          mean_area = 0
          total_area = 0
          min_area = 0
        
        return min_area, max_area, mean_area, total_area        
    
def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

def Image2DConvert(img):
    """
    Convert an RGB image to grayscale if:
    - Only one channel is filled (non-zero).
    - Multiple channels are filled but all have identical pixel values.
    Raise an error if multiple filled channels differ in pixel values.
    
    If input is already grayscale, return as is.
    """
    if img.ndim == 2:
        print("Already grayscale")
        return img
    elif img.ndim == 3:
        filled_channels = np.where(np.any(img != 0, axis=(0, 1)))[0]
        
        if len(filled_channels) == 0:
            raise ValueError("Image has no filled RGB channels.")
        elif len(filled_channels) == 1:
            return img[:, :, filled_channels[0]]
        else:
            # Multiple channels filled, check if they have identical pixel values
            # Reshape to (H*W, C) for filled channels
            filled_pixels = img[:, :, filled_channels].reshape(-1, len(filled_channels))

            if not np.all(np.equal(filled_pixels, filled_pixels[:, [0]]).all(axis=1)):
                raise ValueError("Filled RGB channels differ in pixel values. Choose one channel.")

            return img[:, :, filled_channels[0]]
    else:
        raise ValueError("Input must be a 2D grayscale or 3D RGB image.")

def extract(path_with_images, dict_feat, preprocess, objects_num, obj_class, preproc, preproc_values, enhance, normalize): #features is dict
    print('Extraction started')

    if 'meanvar' in preproc:
        mean = preproc_values['mean']
        std = preproc_values['std']

    if 'pad' in preproc:
        max0 = preproc_values['max0']
        max1 = preproc_values['max1']
     
    data = []
    
    # Create new key in dictionary with features names
    dict_feat['feature_names'] = []
    feat_idx = 0
    for n in dict_feat['features']:
        innerlist = []
        for i in range(dict_feat['feat_num'][feat_idx]):
            innerlist.append(n.class_name() + '_#%02d' % (i+1) )
        dict_feat['feature_names'].append(innerlist)
        dict_feat[n.class_name()] = np.zeros((objects_num, dict_feat['feat_num'][feat_idx]))
        feat_idx += 1
    lst = os.listdir(path_with_images)
    lst.sort()
    # Load images from path_with_images and collect all required features
    for p, files in enumerate(tqdm(lst)):
        cell = {}
        
        img = Image.open(os.path.join(path_with_images, files))
        
        if enhance:   
            enhancer = ImageEnhance.Brightness(img)
            br = calculate_brightness(img)
            base_br = 0.09
            factor = base_br / br            
            
            img = enhancer.enhance(factor)
            #print(calculate_brightness(img))
             
        img = np.array(img)
        img = Image2DConvert(img)

        if normalize:
            #print('Normalization')
            img = (img - img.min()) / (img - img.min()).max()
            img *= 255
            img = img.astype(np.uint8)

        if 'meanvar' in preproc:
            img = preprocess.mean_var_norm(img, mean, std)
            img = (img - img.min()) / (img - img.min()).max()
            img *= 255
            img = img.astype(np.uint8)
        if 'pad' in preproc:
            img = preprocess.padding(img, max0, max1)
        
        ###
        mask_1 = img                                       
        mask_2 = mask_1                                    
        m_val = np.mean(mask_1[mask_1 > 0])                 
        mask_2[mask_2 == 0] = m_val                         
        mask_2 = mask_2 - m_val                            
        mask_2[mask_2 < 0] = 0                             
        img = mask_2                                       
        img = img / img.max()                              
        img *= 255                                         
        img = img.astype(np.uint8)                       
        ###

        for idx, n in enumerate(dict_feat['features']):
            dict_feat[n.class_name()][idx] = n(img)
            
            for f in range(len(dict_feat['feature_names'][idx])): 
                cell.update({str(dict_feat['feature_names'][idx][f]): dict_feat[n.class_name()][idx][f]})

        cell.update({'class_codes': dict_feat[obj_class]})
#         if num_imgs is not None:
        if p==objects_num:
            break;

        data.append(cell)
        
    return data
    

