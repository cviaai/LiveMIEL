
import os
import re
import numpy as np

def batch_mean(X_y_batch):
    """Compute mean features and label for a batch of objects."""
    features_num = X_y_batch.shape[1] - 1
    mean_features = np.mean(X_y_batch[:, :features_num], axis=0)
    label = X_y_batch[0, features_num]
    return mean_features, label

def image_groups(files):
    """
    Group filenames based on the original image name.
    Returns a list of group sizes.
    """
    groups = []
    ref_img_match = re.match(r'(.*)_', files[0])
    ref_img = ref_img_match.group(1) if ref_img_match else ''
    counter = 0

    for filename in files:
        cur_match = re.match(r'(.*)_', filename)
        cur_raw_img = cur_match.group(1) if cur_match else ''
        if cur_raw_img == ref_img:
            counter += 1
        elif cur_raw_img > ref_img:
            groups.append(counter)
            ref_img = cur_raw_img
            counter = 1
    groups.append(counter)
    return groups

def features_averaging(X, y, directories, mode='by_dir', N=False, N_set=False):
    """
    Features averaging modes:
    - mode='by_dir': average over N objects within each directory.
    - mode='by_image': average over objects belonging to the same original image, extracted from filenames.

    Parameters:
    - X: features (array-like)
    - y: labels (array-like)
    - directories: list of directory paths
    - N: number of objects to average per directory (if mode='by_dir')
    - N_set: list of N values per directory (if mode='by_dir')
    - mode: 'by_dir' or 'by_image'
    """
    features_num = X.shape[1]
    y = np.array(y)
    X = np.array(X)
    X_y = np.hstack((X, y.reshape(-1, 1)))

    center_vectors = []
    labels = []

    start_idx = 0

    for dir_idx, directory in enumerate(directories):
        files = sorted(os.listdir(directory))
        total_objects = len(files)

        if mode == 'by_dir':
            if N:
                N_local = N
            elif N_set:
                N_local = N_set[dir_idx]
            else:
                raise ValueError('Averaging value N should be set for all classes or per class in N_set')

            for start_in_dir in range(0, total_objects, N_local):
                end_in_dir = min(start_in_dir + N_local, total_objects)
                global_start = start_idx + start_in_dir
                global_end = start_idx + end_in_dir

                batch = X_y[global_start:global_end, :]
                mean_feat, label = batch_mean(batch)
                center_vectors.append(mean_feat)
                labels.append(label)
            start_idx += total_objects

        elif mode == 'by_image':
            groups = image_groups(files)
            for count in groups:
                stop_idx = start_idx + count
                group = X_y[start_idx:stop_idx, :]
                mean_feat, label = batch_mean(group)
                center_vectors.append(mean_feat)
                labels.append(label)
                start_idx = stop_idx
        else:
            raise ValueError("Mode should be 'by_dir' or 'by_image'")

    return center_vectors, labels
