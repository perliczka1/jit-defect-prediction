# How to reproduce the results?

## Creating conda environment
```
   conda env create -f environment.yml
   conda activate jit-def-pred
   ipython kernel install --user --name=jit-def-pred
```
## Getting the data 
### By your own

1. Clone needed repositories:
* [Openstack](https://github.com/openstack/openstack)
* [Qt](https://github.com/qt)  

```
cd jit-defect-prediction/data/repositories
git clone git@github.com:openstack/openstack.git

cd qt
awk '{ system("git clone git@github.com:qt/"$1".git") }' qt_repositories
```


2. Download files changed by commits:
```
cd jit-defect-prediction
conda activate jit-def-pred
python src/preparation/download_data.py --project openstack
python src/preparation/download_data.py --project qt
```

You can also run test to check if the script works correctly:
```
python src/preparation/download_data.py --project test
```

### Downloading already prepared data
Copy data (around 15GB) from [Google Drive](https://drive.google.com/open?id=18IPjzqOSpJAjI3UIXTaZDNFKDg6xD-dw).
and extract them to `jit-defect-prediction/data` directory.

### Fitting baseline models

How to fit and search for parameters of a baseline model?
```
cd jit-defect-prediction
conda activate jit-def-pred
python src/models/baseline.py --help
```

Results can be found in directory data/models

# Project information
[Google sheet with list of relevant publications, ideas etc.](https://docs.google.com/spreadsheets/d/1K2gpc3aG_N795fbHYZnrsNHq-P11K_WbJrcslj_35gc/edit?usp=sharing)
