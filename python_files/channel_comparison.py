import os
import numpy as np
import shutil
from pathlib import Path

def load_channel_info(dir_info):
    """
    Load labels, coordinates, and number of segments from directories (channels).

    - dir_info: path to files with labels, coordinates, and number of segments
    """
    labels_list = []
    coords_list = []
    n_segm_list = []

    for dir_path in dir_info:
        print(dir_path)
        labels_list.append(np.load(os.path.join(dir_path, 'labels.npy')))
        coords_list.append(np.load(os.path.join(dir_path, 'coordinates.npy')))
        n_segm_list.append(np.load(os.path.join(dir_path, 'number_segmented.npy')))

    return labels_list, coords_list, n_segm_list

def get_object_centers(coords):
    """
    Calculate the center (mean x and y) for each segmented object.

    - coords: ndarray of shape (num_objects, 4) with columns [x_min, x_max, y_min, y_max]
    """
    centers = np.empty((coords.shape[0], 2))
    centers[:, 0] = (coords[:, 0] + coords[:, 1]) / 2  # x center
    centers[:, 1] = (coords[:, 2] + coords[:, 3]) / 2  # y center
    return centers

def compare_objects_by_distance(labels_list, coords_list, n_segments_list):
    """
    Compare objects between channels based on coordinate proximity.

    - labels_list: list of labels for each channel
    - coords_list: list of coordinates for each channel
    - n_segments_list: list of number of segments for each channel
    """
    # Initialize comparison result
    labels_comparison = []

    centers_list = [get_object_centers(coords) for coords in coords_list]
    main_labels = labels_list[0]
    main_centers = centers_list[0]
    main_n = n_segments_list[0]

    # Loop over channel pairs
    for ch_idx in range(1, len(labels_list)):
        curr_labels = labels_list[ch_idx]
        curr_centers = centers_list[ch_idx]
        curr_n = n_segments_list[ch_idx]

        main_idx = 0
        curr_idx = 0

        # List to store comparison results for current pair
        labels_compared = []

        for i in range(main_n.shape[0]):
            main_start = main_idx
            main_end = main_idx + main_n[i]

            curr_start = curr_idx
            curr_end = curr_idx + curr_n[i]

            # Check for empty segments
            if main_n[i] == 0 or curr_n[i] == 0:
                labels_compared.extend([0] * main_n[i])
                main_idx = main_end
                curr_idx = curr_end
                continue

            # Compute pairwise differences
            diff_x = main_centers[main_start:main_end, 0][:, np.newaxis] - curr_centers[curr_start:curr_end, 0][np.newaxis, :]
            diff_y = main_centers[main_start:main_end, 1][:, np.newaxis] - curr_centers[curr_start:curr_end, 1][np.newaxis, :]
            dist_sq = diff_x ** 2 + diff_y ** 2

            # Calculate mean distances for threshold
            segment_coords = coords_list[ch_idx][curr_start:curr_end]
            mean_x = np.mean(((segment_coords[:, 1] - segment_coords[:, 0]) / 2)**2)
            mean_y = np.mean(((segment_coords[:, 3] - segment_coords[:, 2]) / 2)**2)
            threshold = (mean_x + mean_y) / 2

            # Masks based on distances
            mask_x = (dist_sq - dist_sq.min(axis=1, keepdims=True))
            mask_x[mask_x == 0] = -1
            mask_x[mask_x >= 0] = 0
            mask_x = mask_x * (-1) * dist_sq
            mask_x[mask_x >= threshold] = 0
            mask_x[mask_x > 0] = 1

            mask_y = (dist_sq - dist_sq.min(axis=0, keepdims=True))
            mask_y[mask_y == 0] = -1
            mask_y[mask_y >= 0] = 0
            mask_y = mask_y * (-1) * dist_sq
            mask_y[mask_y >= threshold] = 0
            mask_y[mask_y > 0] = 1

            # Compute overlapping labels
            overlap_labels = np.sum(mask_x * mask_y * curr_labels[curr_start:curr_end], axis=1)
            labels_compared.extend(overlap_labels)

            main_idx = main_end
            curr_idx = curr_end

        # Aggregate results
        if len(labels_comparison) == 0:
            labels_comparison = np.array(labels_compared)
        else:
            labels_comparison = labels_comparison * np.array(labels_compared)

    return labels_comparison

def main(directories, save_directory=None):
    """
    Main function to compare objects across channels and save selected objects.
    - Loads data from each channel
    - Performs comparison based on coordinate proximity
    - Saves objects in main channel that have matches in other channels

    - directories: list of paths to the data channels
    - save_directory: path to store compared objects
    """
    # Load data for each channel
    labels_list = []
    coords_list = []
    n_segments_list = []

    info_dirs = [d.replace('nuclei_images', 'saved_info_single_cells', 1) for d in directories]

    labels_list, coords_list, n_segments_list = load_channel_info(info_dirs)

    # Perform comparison
    labels_comparison = compare_objects_by_distance(
        labels_list, coords_list, n_segments_list
    )

    # Load raw image names from main channel
    raw_img_names = np.load(os.path.join(info_dirs[0], 'raw_img_names.npy'))

    # Set source directory
    source_directory = directories[0].replace('nuclei_images', 'single_nuclei_images', 1)

    # Set save directory
    if save_directory is None:
        save_directory = info_dirs[0]
    os.makedirs(save_directory, exist_ok=True)

    # Map main labels to their corresponding object files
    main_labels = labels_list[0]
    main_dir = directories[0]

    # Loop through all labels in the main channel
    start_idx = 0
    for i, n_seg in enumerate(n_segments_list[0]):
        stop_idx = start_idx + n_seg
        cur_raw_img = raw_img_names[i]

        # Get the detected labels for this image
        cur_labels = main_labels[start_idx:stop_idx]
        # Corresponding comparison labels indicating matches
        matched_labels = labels_comparison[start_idx:stop_idx]

        # For each label in the current image
        for label_idx, label in enumerate(cur_labels):
            # If label has a match (labels_comparison > 0)
            if labels_comparison[start_idx + label_idx] > 0:
                filename = f'nucleus_{cur_raw_img[:-4]}_{label:02d}.tif'
                src_path = os.path.join(source_directory, filename)
                dst_path = os.path.join(save_directory, filename)

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    #print(f"Copied: {src_path} to {dst_path}")
                else:
                    print(f"File not found: {src_path}")
        start_idx = stop_idx

#################################

def move_common_files(source_dir, comparison_dir, backup_dir, create_subdirs=True):
    """
    Move files from source_dir to backup_dir if they exist in comparison_dir.
    
    Args:
        - source_dir (str): Directory from which to remove/move files
        - comparison_dir (str): Directory containing files to compare against
        - backup_dir (str): Directory where matched files will be moved
        - create_subdirs (bool): Whether to create backup subdirectories for organization
    """
    # Convert to Path objects for easier handling
    source_path = Path(source_dir)
    comparison_path = Path(comparison_dir)
    backup_path = Path(backup_dir)
    
    # Validate directories exist
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    if not comparison_path.exists():
        raise ValueError(f"Comparison directory does not exist: {comparison_dir}")
    
    # Create backup directory if it doesn't exist
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Get all filenames from comparison directory
    comparison_files = set()
    for item in comparison_path.iterdir():
        if item.is_file():
            comparison_files.add(item.name)
    
    print(f"Found {len(comparison_files)} files in comparison directory")
    print(f"Looking for matching files in: {source_dir}")
    
    # Counter for moved files
    moved_count = 0
    
    # Process files in source directory
    for item in source_path.iterdir():
        if item.is_file() and item.name in comparison_files:
            try:
                # Determine target path
                if create_subdirs:
                    # Create subdirectory in backup named after source directory
                    source_dir_name = source_path.name
                    target_subdir = backup_path / source_dir_name
                    target_subdir.mkdir(exist_ok=True)
                    target_path = target_subdir / item.name
                else:
                    target_path = backup_path / item.name
                
                # Move the file
                shutil.move(str(item), str(target_path))
                moved_count += 1
                print(f"Moved: {item.name} -> {target_path}")
                
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
    
    print(f"\nOperation complete! Moved {moved_count} files to backup directory.")