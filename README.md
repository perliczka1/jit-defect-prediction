
# How to reprocude the results?

## Creating conda environment:
```
   conda env create -f environment.yml
   conda activate jit-def-pred
```
## Getting the data by your own

1. Clone needed repositories:
* [Openstack](https://github.com/openstack/openstack)
* [Qt](https://github.com/qt)  

```
cd smart-jit/data/repositories
git clone git@github.com:openstack/openstack.git

cd qt
awk '{ system("git clone git@github.com:qt/"$1".git") }' qt_repositories
```

2. Download files changed by commits:
```
cd smart-jit
conda activate smart-jit
python src/preparation/download_data.py --project openstack
python src/preparation/download_data.py --project qt
```

You can also run test to check if the script works correctly:
```
python src/preparation/download_data.py --project test
```



TODO
* Ignore comments only
* Ignore new lines
* Ignore files that are not code (.py or .cpp)