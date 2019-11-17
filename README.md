
# create conda environment:
conda env create -f environment.yml
conda activate smart-jit

# download data: 
python src/preparation/download_data.py --project openstack
python src/preparation/download_data.py --project qt

TODO
* Ignore comments only
* Ignore new lines
* Ignore files that are not code (.py or .cpp)