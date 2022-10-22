import os
from typing import Iterable, IO, Optional


def makedirs_if_new(path: str) -> bool:
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def close_files(files: Iterable[Optional[IO]]) -> None:
    for f in files:
        if f is not None:
            f.close()


def create_writable_file_if_new(folder: str, name: str):
    makedirs_if_new(folder)
    results_file = os.path.join(folder, name + '.txt')
    if os.path.isfile(results_file):
        return None
    return open(results_file, 'w')


def get_best_ckpt(models_folder, class_name: str):
    for folder in models_folder.iterdir():
        if ("_" + class_name) not in folder.name:
            continue

        ckpt_folder = folder / "version_0" / "checkpoints"
        for ckpt_file in sorted(ckpt_folder.iterdir()):
            if "last" in ckpt_file.name:
                continue
            return ckpt_file


def folder_name_from_params(max_edge_length: int, deltas: bool, polar: bool, no_sameframe: bool, max_edge_distance_multiplier: int = 1) -> str:
    name = ""
    name += f"_len{'full' if max_edge_length==-1 else str(max_edge_length)}"
    name += f"_{'' if deltas else 'no'}delta{'polar' if polar else 'cartesian'}"
    name += f"_{'no' if no_sameframe else ''}sameframe"
    if max_edge_distance_multiplier != 1:
        name += f"_dist{max_edge_distance_multiplier}"
    return name
