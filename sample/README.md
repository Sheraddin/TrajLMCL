# Dataset Samples (TrajLMCL)

This directory is intended to store the `.h5` data files required for training and evaluating the model. Due to the large file sizes (approximately 9 GB), these datasets are hosted externally and are not included directly in the Git repository.

## Download Link
You can download the full dataset from Google Drive here:
[TrajLMCL Dataset Samples](https://drive.google.com/drive/folders/1ccPIEXO1NxgTscSKdJdbzbcXNN9sLv3_?usp=sharing)

## File Descriptions
* **small_chengdu_Dx.h5**: Trajectory samples from Chengdu (versions D0-D4).
* **small_xian_Dx.h5**: Trajectory samples from Xi'an (versions D0-D4).
* **small_chengdu.h5 / small_xian.h5**: Base/merged dataset files.

## Setup Instructions
1. Follow the Google Drive link provided above.
2. Download all the `.h5` files.
3. Place the downloaded files into this `sample/` directory.
4. Ensure the filenames match the list above so that the `dataloader` scripts can recognize them.

---
*Note: Local `.h5` files are ignored by Git (via `.gitignore`) to prevent accidental uploads of large binary data.*
