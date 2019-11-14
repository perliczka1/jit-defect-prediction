import os
from typing import List
import git


def get_list_of_touched_files(repo: git.Repo, commit: git.Commit) -> List[str]:
    parent_commit = commit.parents[0]
    diffs = commit.diff(parent_commit)
    files = [diff.a_path or diff.b_path for diff in diffs]  # if a file was added we get null as a_path
    return files


def save_file_as_in_commit(file_path: str, repo: git.Repo, commit: git.Commit, save_directory: str) -> None:
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_name_from_path = file_path.replace("/", "_")
    save_path = os.path.join(save_directory, file_name_from_path)
    file_str = repo.git.show('{}:{}'.format(commit.hexsha, file_path))
    with open(save_path, 'w') as writer:
        writer.write(file_str)


def save_file_before_and_after(file_path: str, repo: git.Repo, commit: git.Commit, path_to_save) -> None:
    save_directory_for_commit = os.path.join(path_to_save, commit.hexsha)
    save_file_as_in_commit(file_path, repo, commit, os.path.join(save_directory_for_commit, 'after'))
    parent_commit = commit.parents[0]
    save_file_as_in_commit(file_path, repo, parent_commit, os.path.join(save_directory_for_commit, 'before'))


def process_commits(repository_paths: List[str], commit_ids: List[str], path_to_save: str) -> None:
    repos = [git.Repo(r_p) for r_p in repository_paths]
    repo = repos[0]
    commits = [repo.commit(c_i) for c_i in commit_ids]
    for c in commits:
        files = get_list_of_touched_files(repo, c)
        for f in files:
            save_file_before_and_after(f, repo, c, path_to_save)


if __name__ == '__main__':
    # print(save_files_before_and_after("../../data/repositories/swift", "ffadcb78c70f2d4e092eaaf8ade2634f9f2e9542"))
    repo_path = "../../data/repositories/swift"
    commit_id = "ffadcb78c70f2d4e092eaaf8ade2634f9f2e9542"
    process_commits([repo_path], [commit_id], "../../data/files/openstack")
    # print(save_file_before_and_after("swift/account/reaper.py",
    #                                  repo,
    #                                  commit,
    #                                  "../../data/files/openstack"))

