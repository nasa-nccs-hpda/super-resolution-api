
# super-resolution-api

Super Resolution Application Programming Interface (API) for Weather/Climate Data Framework.

## Environment

If mamba is not available, install [miniforge](https://github.com/conda-forge/miniforge).
Execute the following to set up a conda environment for super-resolution-api:

    >  mamba create -n sres python=3.11
    >  mamba activate sres
    >  mamba install -c conda-forge dask scipy xarray netCDF4 ipywidgets=7.8 jupyterlab=4.0 jupyterlab_widgets ipykernel=6.29 ipympl=0.9 ipython=8.26
    >  mamba install -c pytorch -c nvidia -c conda-forge litdata pytorch lightning lightning-utilities torchvision torchaudio pytorch-cuda cuda-python
    >  pip install parse  nvidia-dali-cuda120
    >  pip install hydra-core --upgrade
    >  ipython kernel install --user --name=sres

## Setup

Execute the following to install and setup the super-resolution-api framework.

    > git clone https://github.com/nasa-nccs-hpda/super-resolution-api.git
    > cd super-resolution-api/
    > export PYTHONPATH=.:./super-resolution-climate:$PYTHONPATH

## Configuration

This project uses [hydra](https://hydra.cc) for workflow configuration.  All configuration files are found in the super-resolution-api/super-resolution-climate/config directory.

Each workflow configuration is composed of several sections, each with a separate config file. For example, in the sample script [train-rcan-swot-2.2v.py](./scripts/train-rcan-swot-2.2v.py), 
the *configuration* dict specifies the name of the config file to be used for each section, i.e. the *task* section is configured with the file [config/task/swot-2.2v.yaml](./config/task/SSS_SST-tiles-48.yaml). 
The *ccustom* dict is used to override individual config values.  The *cname* parameter specifies the name of the root config file (e.g. [config/sres.yaml](./config/sres.yaml) )

## Parameters

| Parameter | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |

## Training

The scripts under *super-resolution-api/scripts/train* are used to train various super-resolution networks with various configurations. The notebook 
[super-resolution-api/notebooks/plot_training.ipynb](./notebooks/plot_training.ipynb) is used to display a plot of 
loss vs. epochs for the configured training instance.

## Inference

The scripts under *super-resolution-api/scripts/inference* are used to run inference for the trained super-resolution networks. 

