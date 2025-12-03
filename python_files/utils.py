import os

def create_result_directories(raw_data_dir):
    """
    Create directories to store segmentation results.

    Args:
    - raw_data_dir (str): Path to the raw data directory inside nuclei_images.
    """
    _data_dir = os.path.abspath(raw_data_dir)
    segm_data_dir = raw_data_dir.replace('nuclei_images', 'single_nuclei_images', 1)
    info_data_dir = raw_data_dir.replace('nuclei_images', 'saved_info_single_cells', 1)

    for dir in [segm_data_dir, info_data_dir]:
        try:
            os.makedirs(dir, exist_ok=True)
            print(f"Created directory: {dir}")
        except OSError as error:
            print(f"Failed to create directory: {dir}")
                
    return segm_data_dir, info_data_dir

