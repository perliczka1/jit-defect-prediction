import os
from typing import List, Dict
import git
import pandas as pd
import shutil
import time
from multiprocessing import Pool
from collections import ChainMap, OrderedDict
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from utils import input_dataset_path, repo_paths_for_project, file_with_changes_path, file_with_summary_path


def save_file_as_in_commit(file_path: str, repo: git.Repo, commit: git.Commit,
                           save_directory: str, file_path_to_save: str, not_present: bool) -> None:
    """
    Save version of the file from a specific commit. If file was not present save empty string.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_name_from_path = file_path_to_save.replace("/", "_")
    save_path = os.path.join(save_directory, file_name_from_path)
    if not_present:
        file_str = ''
    else:
        try:
            show_input = '{}:{}'.format(commit.hexsha, file_path)
            file_str = repo.git.show(show_input)
        except git.exc.GitCommandError as exp:
            print(f"Error when executing git show {show_input}")
            file_str = ''
    with open(save_path, 'w', encoding="utf-8", errors="surrogateescape") as writer:
        writer.write(file_str)


def get_summary_for_commit(commit_id: str, diffs: git.diff.DiffIndex) -> pd.DataFrame:
    CHANGE_TYPES = ['A', 'D', 'R', 'M', 'T']
    change_counts = OrderedDict([(ct, 0) for ct in CHANGE_TYPES])
    for diff in diffs:
        change_counts[diff.change_type] += 1
    return pd.DataFrame(data = [[commit_id, len(diffs)] + list(change_counts.values())],
                        columns=['commit_id', 'files_changed'] + CHANGE_TYPES)


def save_files_before_and_after(repo_path: str, commit_id: str, path_to_save: str) -> None:
    """
    Possible modifications of the files:
        ’A’ for added paths
        ’D’ for deleted paths
        ’R’ for renamed paths
        ’M’ for paths with modified data
        ’T’ for changed in the type paths
    """
    repo = git.Repo(repo_path)
    commit = repo.commit(commit_id)
    save_directory_for_commit = os.path.join(path_to_save, commit.hexsha)
    if len(commit.parents) < 1:
        parent_commit = commit # if there is no parent then we assume no change but we print this case to investigate it
        print("No parent found for commit: " + commit_id)
        no_parent = True
    else:
        parent_commit = commit.parents[0]
        no_parent = False
    diffs = parent_commit.diff(commit, ignore_blank_lines=True, ignore_all_space=True)
    for diff in diffs:
        not_present_before = no_parent or (diff.change_type == 'A') # if there is no parent then the whole repo was just initialized by the commit
        not_present_after = diff.change_type == 'D'
        save_file_as_in_commit(diff.a_path, repo, parent_commit, os.path.join(save_directory_for_commit, 'before'),
                               diff.b_path, not_present_before)
        save_file_as_in_commit(diff.b_path, repo, commit, os.path.join(save_directory_for_commit, 'after'),
                               diff.b_path, not_present_after)
    return get_summary_for_commit(commit_id, diffs)


def process_commits(commits_to_repo_path: Dict[str, str], path_to_save: str) -> pd.DataFrame:
    shutil.rmtree(path_to_save, ignore_errors=True)
    arguments = [(r, c, path_to_save) for c, r in commits_to_repo_path.items()]
    pool = Pool(8)
    summary_dfs = pool.starmap(save_files_before_and_after, arguments)
    return pd.concat(summary_dfs)


def commit_ids_from_csv(dataset_path: str) -> List[str]:
    df = pd.read_csv(dataset_path)
    return df['commit_id'].tolist()


def match_commits_to_repo(commit_ids: List[str], repo_path: str) -> List[Dict[str, str]]:
    commits_to_repo_path = {}
    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        return {}
    for c_i in commit_ids:
        try:
            repo.commit(c_i)
            commits_to_repo_path[c_i] = repo_path
        except Exception:
            continue
    return commits_to_repo_path


def match_commits_to_repos(commit_ids: List[str], repo_paths: List[str]) -> List[Dict[str, str]]:
    pool = Pool(8)
    arguments = [(commit_ids, repo_path) for repo_path in repo_paths]
    results = pool.starmap(match_commits_to_repo, arguments)
    commits_to_repo_path = dict(ChainMap(*results))
    success_rate = len(commits_to_repo_path) / len(commit_ids) * 100
    print(f'Successfully processed {success_rate} %')
    return commits_to_repo_path


def check_summary(input_dataset_path: str, summary_path:str) -> None:
    input = pd.read_csv(input_dataset_path)
    summary = pd.read_csv(summary_path)
    joined = input.merge(summary, how='left', on='commit_id')
    missing_rows = joined['files_changed'].isnull().mean()
    joined['files_changed_without_added'] = joined['files_changed'] - joined['A']
    incorrect_number_of_modified_files = (joined['files_changed_without_added'] != joined['nf']).mean()
    print('Summary:')
    print(summary.describe())
    print('Checks: ')
    print(f'Percentage of missing rows: {missing_rows}, percentage of different numbers of modified files: {incorrect_number_of_modified_files}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='Name of the project to process', choices=['test', 'openstack', 'qt'], required=True)
    args = parser.parse_args()
    start = time.time()
    repo_paths = repo_paths_for_project(args.project)
    commit_ids = commit_ids_from_csv(input_dataset_path(args.project))
    print('Matching commits to repositories')
    commits_to_repo_path = match_commits_to_repos(commit_ids, repo_paths)
    print('Processing commits')
    df = process_commits(commits_to_repo_path, file_with_changes_path(args.project))
    df.to_csv(file_with_summary_path(args.project), index=False)
    end = time.time()
    print('Finished in ', time.strftime('%H:%M:%S', time.gmtime(end - start)))
    check_summary(input_dataset_path(args.project), file_with_summary_path(args.project))



