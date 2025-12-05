# **LiveMIEL+: Processing Microscopy Images of Epigenetic Landscape**
> Distinguish between epigenetic states in living cells directly from raw fluorescence microscopy data using a complete image processing pipeline.
>
## **Installation**

LiveMIEL+ is configured for immediate use in Google Colab with Google Drive integration.

### **Setup Steps**

1. **Upload to Google Drive:**
   - Upload the entire `LiveMIEL+` package to your Google Drive
   - Ensure `python_files/` and `Segmentation_features_extraction_clustering.ipynb` are in the project root

2. **Prepare Data Directory:**
   - Inside the `LiveMIEL+` directory, create a folder `nuclei_images`
   - Add your microscopy images or download test data [here](https://drive.google.com/drive/folders/1tswl-JbL0TjlC_Y81enjiclf2Bvz0HJP?usp=sharing)

3. **Run Analysis:**
   - Open `Segmentation_features_extraction_clustering.ipynb` in Google Colab
   - Mount your Google Drive when prompted
   - Install dependencies (automated in notebook)
   - Execute the notebook cells

### **Technical Details**


**Supported Image Formats**
- **Grayscale 8-bit** (single channel intensity)
- **One-channel RGB** (only one channel contains data)
- **Uniform RGB** (all three channels have identical values)

**Recommended Format:** TIFF (high-resolution, lossless compression - common in microscopy)

**Dependencies:** All required packages are listed in `requirements.txt` and automatically handled in the notebook's *Libraries import* section.

## **Usage Example**

The notebook `Segmentation_features_extraction_clustering.ipynb` provides a complete walkthrough of LiveMIEL+ applied to a cell-cycle study dataset. You can download the test data [here](https://drive.google.com/drive/folders/1tswl-JbL0TjlC_Y81enjiclf2Bvz0HJP?usp=sharing).

**Dataset Description:**
The dataset consists of fluorescent microscopic images of HEK293 cells from a 24-hour time-lapse experiment, with images captured every 15 minutes.

- **Microscope Settings:** 20x magnification, 1410 x 1200 px resolution.
- **Channels:** Images were collected in three channels (Red, Yellow, Blue), each stored in a separate directory named after its color.
- **Biological Context:**
  - **Red Channel:** Epigenetic landscape of methylation mark H3K9me3.
  - **Yellow & Blue Channels:** Cell cycle phase indicators.
    - Nuclei in **Yellow only**: S phase.
    - Nuclei in **Blue only**: G1 phase.
    - Nuclei in **both Yellow and Blue**: G2/M phase.

## Tunable Parameters

The segmentation pipeline is highly configurable to adapt to different image qualities and magnifications. It consists of three stages:

1.  **Bandpass Segmentation:** `(image * big_gaussian_kernel) - coeff x (image * small_gaussian_kernel) > thresh`
2.  **Watershed Segmentation:** Separates adjacent or touching nuclei.
3.  **False Positive Removal:** Filters out objects that do not correspond to true nuclei.

### Core Parameters

| Parameter | Description | Typical Range (60x, 1360x1024 px) |
| :--- | :--- | :--- |
| **`lowSigm`** | Std. dev. of the `small_gaussian_kernel`. Smoothes noise and defines object detail. Lower values preserve more detail; higher values smooth outlines. <br><br> <img src="./figures/lowSigm.png" style="width:350px; height:auto;">  | `[5, 15]` |
| **`highSigm`** | Std. dev. of the `big_gaussian_kernel`. Removes background. Lower values remove more background fragments. <br><br><img src="./figures/highSigm.png" style="width:350px; height:auto;"> | `[20, 70]` |
| **`thresh`** | Intensity threshold for the bandpass filter. Pixels above this value are considered potential objects. | `[0.003, 0.05]` |
| **`FalsePositBrightness_k`** | Brightness coefficient `k`. Objects with mean fluorescence < `k * average_image_fluorescence` are removed. <br><br><img src="./figures/FalsePositBrightness_k.png" style="width:550px; height:auto;"> | `[1.0, 2.5]` |
| **`MinNucleusArea`** | Minimum area (in pixels). Objects smaller than this are discarded. | `[500, 1500]` |

### Scaling Parameters for Different Magnifications/Resolutions

The parameters `lowSigm`, `highSigm`, and `MinNucleusArea` **scale linearly** with the image dimensions.

**Example:** For a **60x** image with a resolution of **4080 x 3072 px** (3x larger in each dimension than the base 1360x1024):

- **Scale Factor:** `(4080 / 1360) = 3`
- **Area Scale Factor:** `3 * 3 = 9`

| Parameter | Scaled Value / Range |
| :--- | :--- |
| **`lowSigm`** | `[5*3, 15*3]` = `[15, 45]` |
| **`highSigm`** | `[20*3, 70*3]` = `[60, 210]` |
| **`MinNucleusArea`** | `1000 * 9` = `9000` |


**Tips for Parameter Tuning:**
- Start with the scaled base values and adjust incrementally
- Use the example images above as visual references for expected outcomes


