import numpy as np
import math
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
from PIL import Image

from skimage.segmentation import watershed
from scipy import ndimage
from scipy.ndimage import grey_dilation, generate_binary_structure, \
        maximum_filter, minimum_filter, \
        gaussian_filter, \
        center_of_mass, label

from collections import Counter

def bandPassSegm(img, lowSigm, highSigm, coeff, thresh):

    img = img - img.min()
    img =img/ img.max()

    img_sm = gaussian_filter(img, sigma = lowSigm, mode = 'reflect', radius = int((np.ceil(3*lowSigm)-1)/2))
    img_bg = gaussian_filter(img, sigma = highSigm, mode = 'reflect', radius = int((np.ceil(3*highSigm)-1)/2))

    img_sample = img_sm - img_bg * coeff

    im_mask = img_sample > thresh

    return im_mask.astype(float)

######GALA FUNCTIONS###################################
def complement(a):
    return a.max()-a

def hminima(a, thresh):

    maxval = a.max()
    ainv = maxval - a

    return maxval - morphological_reconstruction(ainv-thresh, ainv)

def morphological_reconstruction(marker, mask, connectivity=1):

    sel = generate_binary_structure(marker.ndim, connectivity)
    marker = reconstruction(marker, mask, footprint = sel)

    return marker

def regional_minima(a, connectivity=1):
    """Find the regional minima in an ndarray."""
    values = np.unique(a)
    delta = (values - minimum_filter(values, footprint=np.ones(3)))[1:].min()
    marker = complement(a)
    mask = marker+delta
    return (marker == morphological_reconstruction(marker, mask, connectivity)).astype(float)
#######################################################

def watershedSegm(mask):

    dist = -ndimage.distance_transform_edt(mask)
    local   = hminima(dist, 2)
    local   = regional_minima(local, 8)
    markers = ndimage.label(local)[0]
    labels = watershed(dist, markers, mask=mask)

    return labels

def remove_false_positives(img, labels, FalsePositBrightness_k, MinNucleusArea):
    """
    Find and remove all segmented areas with intensity below FalsePositBrightness_k*img.mean
    where img.mean is mean intensity of segmeneted masked image.
    """
    binary_mask = labels.copy()
    binary_mask[binary_mask != 0] = 1

    masked_img = np.multiply(binary_mask, img)
    masked_img_thresholded = masked_img - (FalsePositBrightness_k*img.mean())

    masked_img_thresholded[masked_img_thresholded < 0] = 0
    masked_img_thresholded[masked_img_thresholded != 0] = 1

    new_labels_ = np.multiply(labels, masked_img_thresholded)

    labels[np.isin(labels, np.unique(new_labels_).astype(int)) == False] = 0

    """Find and remove all segmented areas less than MinNucleusArea."""
    labels_tmp = labels.copy()
    labels_tmp = labels_tmp[labels_tmp!=0].flatten()
    labels_dict = Counter(tuple(labels_tmp))
    keys = np.array(list(labels_dict.keys()))
    values = list(labels_dict.values())
    values = np.array(list(map(int, values)))
    keys[values <= MinNucleusArea] = 0

    labels[np.isin(labels, np.unique(keys).astype(int)) == False] = 0

    return labels

def remove_border_objects(labeled_mask):
    """
    Removes segmented objects from the mask if their center of mass is closer to the image edges
    than the average radius of all objects inside the image.
    """
    #labeled_mask, num_objects = label(mask)
    object_labels = np.unique(labeled_mask)
    object_labels = object_labels[object_labels != 0]
    image_h, image_w = labeled_mask.shape

    radii = []
    centers = []

    # Calculate radius and center of mass for each object
    for obj_label in object_labels:#range(1, num_objects + 1):
        obj_mask = (labeled_mask == obj_label)
        # Center of mass
        com = center_of_mass(obj_mask)
        centers.append(com)
        # Compute bounding box
        coords = np.argwhere(obj_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Radius as half of width & height
        radius = 0.5 * max(y_max - y_min + 1, x_max - x_min + 1)
        radii.append(radius)

    if len(radii) == 0:
        return labeled_mask  # No objects to process

    avg_radius = np.mean(radii)

    # Determine distances of object centers to image edges
    centers = np.array(centers)
    distances_to_edges = np.minimum.reduce([
        centers[:, 0],  # distance to top edge
        image_h - centers[:, 0],  # bottom
        centers[:, 1],  # left
        image_w - centers[:, 1]  # right
    ])

    '''# Remove objects close to edges
    mask_cleaned = labeled_mask.copy()
    for idx, dist in enumerate(distances_to_edges):
        if dist < avg_radius:
            mask_cleaned[labeled_mask == (idx + 1)] = 0'''
    
        # Vectorized identification of objects to remove
    objects_to_remove = object_labels[distances_to_edges < avg_radius]
    
    # Remove identified objects using vectorized operation
    mask_cleaned = labeled_mask.copy()
    if len(objects_to_remove) > 0:
        remove_mask = np.isin(labeled_mask, objects_to_remove)
        mask_cleaned[remove_mask] = 0

    return mask_cleaned


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


def main(img, lowSigm, highSigm, coeff, thresh,
         removeFalsePosit=True, FalsePositBrightness_k=1.5, MinNucleusArea=1000,
         removeBorderObjects=True):

    img = Image2DConvert(img)

    assert type(removeFalsePosit)==bool, 'removeFalsePosit must be Bool'
    assert type(removeBorderObjects)==bool, 'removeBorderObjects must be Bool'

    print('Segmenting cells with Banpass filter High Std %2d, Low Std %2d' % (highSigm, lowSigm))
    segmented_image = bandPassSegm(img, lowSigm, highSigm, coeff, thresh)

    print('<========== Performing Watershed segmentation ==========>')
    new_segmented_image = watershedSegm(segmented_image)

    if removeFalsePosit:
        print('<========== Removing false positive segmented areas ==========>')
        new_segmented_image = remove_false_positives(img, new_segmented_image, FalsePositBrightness_k, MinNucleusArea)
    
    if removeBorderObjects:
        print('<========== Removing close to borders segmented areas ==========>')
        new_segmented_image = remove_border_objects(new_segmented_image)

    return new_segmented_image