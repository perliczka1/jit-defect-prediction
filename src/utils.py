import os
from glob import glob
from typing import List


def project_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


def input_dataset_path(project_name: str) -> str:
    return os.path.join(project_path(), 'data', 'input_datasets', f'{project_name}.csv')


def repo_paths_for_project(project_name: str) -> List[str]:
    if project_name == 'test':
        return [os.path.join(project_path(), 'data', 'repositories', 'openstack', "cinder")]
    elif project_name == 'qt':
        path_to_directories = os.path.join(project_path(), 'data', 'repositories', 'qt', "*" )
        return glob(path_to_directories + '/')
    elif project_name == 'openstack':
        path_to_directories = os.path.join(project_path(), 'data', 'repositories', 'openstack', "*" )
        return glob(path_to_directories + '/')
    else:
        raise ValueError('Incorrect project_name')


def file_with_changes_path(project_name: str) -> str:
    return os.path.join(project_path(), 'data', 'files', project_name)


def file_with_summary_path(project_name: str) -> str:
    directory = os.path.join(project_path(), 'data', 'generated_datasets', 'summary')
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.join(directory, f'{project_name}.csv')