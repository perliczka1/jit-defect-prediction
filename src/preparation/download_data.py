import os
from typing import List
import git
import pandas as pd
from utils import project_path, input_dataset_path, repositories_path
import shutil
from tqdm import tqdm
import concurrent.futures
import time
from multiprocessing import Pool

# def list_of_touched_files(repo: git.Repo, commit: git.Commit) -> List[str]:
#     """
#     ’A’ for added paths
#     ’D’ for deleted paths
#     ’R’ for renamed paths
#     ’M’ for paths with modified data
#     ’T’ for changed in the type paths
#     """
#     parent_commit = commit.parents[0]
#     diffs = parent_commit.diff(commit)
#     files = []
#     for diff in diffs:
#         files.append(diff.a_path or diff.b_path)
#     files = [(diff.a_path, diff.b_path for diff in diffs]  # if a file was added we get null as a_path
#     return files


def save_file_as_in_commit(file_path: str, repo: git.Repo, commit: git.Commit,
                           save_directory: str, file_path_to_save: str, not_present: bool) -> None:
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_name_from_path = file_path_to_save.replace("/", "_")
    save_path = os.path.join(save_directory, file_name_from_path)
    if not_present:
        file_str = ''
    else:
        file_str = repo.git.show('{}:{}'.format(commit.hexsha, file_path))
    with open(save_path, 'w', encoding="utf-8", errors="surrogateescape") as writer:
        writer.write(file_str)


def save_files_before_and_after(repo: git.Repo, commit: git.Commit, path_to_save: str) -> None:
    """
        ’A’ for added paths
        ’D’ for deleted paths
        ’R’ for renamed paths
        ’M’ for paths with modified data
        ’T’ for changed in the type paths
    """
    save_directory_for_commit = os.path.join(path_to_save, commit.hexsha)
    parent_commit = commit.parents[0]
    diffs = parent_commit.diff(commit)
    for diff in diffs:
        not_present_before = diff.change_type == 'A'
        not_present_after = diff.change_type == 'D'
        save_file_as_in_commit(diff.a_path, repo, parent_commit, os.path.join(save_directory_for_commit, 'before'),
                               diff.b_path, not_present_before)
        save_file_as_in_commit(diff.b_path, repo, commit, os.path.join(save_directory_for_commit, 'after'),
                               diff.b_path, not_present_after)

def save_files_before_and_after_for_str(repo_path: str, commit_id: str, path_to_save: str) -> None:
    """
        ’A’ for added paths
        ’D’ for deleted paths
        ’R’ for renamed paths
        ’M’ for paths with modified data
        ’T’ for changed in the type paths
    """
    repo = git.Repo(repo_path)
    commit = repo.commit(commit_id)
    save_files_before_and_after(repo, commit, path_to_save)


def process_commits(repository_paths: List[str], commit_ids: List[str], path_to_save: str) -> None:
    shutil.rmtree(path_to_save, ignore_errors=True)
    repos = [git.Repo(r_p) for r_p in repository_paths]
    repo = repos[0]
    commits = []
    commit_ids_ok = []
    errors = []
    for c_i in commit_ids:
        try:
            commits.append(repo.commit(c_i))
            commit_ids_ok.append(c_i)
        except Exception as exc:
            errors.append(exc)
    success_rate = len(commits) / len(commit_ids) * 100
    print(f'Successfully processed {success_rate} %')
    #commit_ids_ok = commit_ids_ok[:100]
    arguments = [(repository_paths[0], c, path_to_save) for c in commit_ids_ok]
    pool = Pool(4)
    pool.starmap(save_files_before_and_after_for_str, arguments)


def commit_ids_from_csv(dataset_path: str) -> List[str]:
    df = pd.read_csv(dataset_path)
    return df['commit_id'].tolist()


if __name__ == '__main__':
    # print(save_files_before_and_after("../../data/repositories/swift", "ffadcb78c70f2d4e092eaaf8ade2634f9f2e9542"))
    start = time.time()
    repo_paths = repositories_path('openstack')
    commit_ids = commit_ids_from_csv(input_dataset_path('openstack'))
    process_commits(repo_paths, commit_ids, "../../data/files/openstack")
    end = time.time()
    print((end - start) / 60)


