import os


def project_path() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def input_dataset_path(project_name: str) -> str:
    if project_name == 'openstack':
        return os.path.join(project_path(), 'data', 'input_datasets', 'openstack.csv')
    elif project_name == 'qt':
        return os.path.join(project_path(), 'data', 'input_datasets', 'qt.csv')
    else:
        raise ValueError('project not found')


def repositories_path(project_name: str) -> str:
    if project_name == 'openstack':
        return [os.path.join(project_path(), 'data', 'repositories', 'swift')]
    elif project_name == 'qt':
        return [os.path.join(project_path(), 'data', 'repositories', 'swift')]
    else:
        raise ValueError('project not found')