# Lightweight Event-based Optical Flow Estimation via Iterative Deblurring

This is the code for the ICRA 2024 submission #2581

## Dependency
Create a conda env and install dependencies by running
```
conda env create --file environment.yml
```

## Download (For Evaluation)
The DSEC dataset for optical flow can be downloaded [here](https://dsec.ifi.uzh.ch/dsec-datasets/download/).
Use script [download_dsec_test.py](download_dsec_test.py) for your convenience.
It downloads the dataset directly into the `DATA_DIRECTORY` with the expected directory structure.
```python
download_dsec_test.py <DATA_DIRECTORY>
```
Once downloaded, create a symbolic link called  `data` pointing to the data directory:
```
ln -s <DATA_DIRECTORY> data/test
```

## Download (For Training)
For training on DSEC, two more folders need to be downloaded:

- Unzip [train_events.zip](https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_events.zip) to data/train_events

- Unzip [train_optical_flow.zip](https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_optical_flow.zip) to data/train_optical_flow

or establish symbolic links under data/ pointing to the folders.

## Download (MVSEC)
To run experiments on MVSEC, additionally download outdoor day sequences .h5 files from https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV
and place the files under data/ or point symbolic links pointing to the data files under data/.

## Run Evaluation

To run eval:
```
cd idnet
conda activate IDNet
python -m idn.eval
```

Change the save directory for eval results in `idn/config/validation/dsec_test.yaml` if you prefer. The default is at `/tmp/collect/XX`.

To switch between models, change the model option in `idn/config/id_eval.yaml` to switch between id model with 1/4 and 1/8 resolution.

To eval TID model, change the function decorator above the main function in `eval.py`.

At the end of evaluation, a zip file containing the results will be created in the save directory, for which you can upload to the DSEC benchmark website to reproduce our results.

## Run Training
To train IDNet, run:
```
cd idnet
conda activate IDNet
python -m idn.train
```

Similarly, switch between id-4x, id-8x and tid models and MVSEC training by changing the hydra.main() decorator in `train.py` and settings in the corresponding .yaml file.

