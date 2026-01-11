# GR00T SVCCA

## 1. Setup

Install basic dependencies.
```bash
conda create -n svcca python=3.10
conda activate svcca
pip install h5py numpy torch scipy
```

Setup GR00T repository
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

If you have already setup the Isaac-GR00T in your machine, move to step 2.


## 2. Caching Model Activations

Extract and cache model activations in HDF5 format for a specific dataset.
Before running the script, set the GR00T_PATH at `line 5` in `extract_layer_features.py`

```bash
bash run_extract_activations.sh
```

This saves model activations to HDF5 files.

## 3. Compute SVCCA

Compare layers between two models with identical architecture:

```bash
python svcca.py --actv_a_path <model1_features.hdf5> --actv_b_path <model2_features.hdf5>
```
