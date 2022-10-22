import argparse
import os
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import ujson as json

from utils.io import makedirs_if_new

MAX_TRACKS_PER_CLASS = int(1e6)  # to make each track globally unique, not only within each class

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str,
                    help="The name of the json results file (without .json extension) - has to be the same for all results that will be merged, e.g. v1.0-trainval_tracking")
parser.add_argument("root_folder", type=str,
                    help="The path to the root folder inside of which result folders are stored, e.g. /storage/runs")
parser.add_argument("folder_substr", type=str,
                    help="A common substring of all folders to merge, e.g. 'nodeltapolar' ")
parser.add_argument("-merged_folder_suffix", type=str, default="",
                    help="The suffix for the merged folder name, e.g. ablation_no_delta")
args = parser.parse_args()

root_folder = Path(args.root_folder)
folders_to_merge = [str(folder) for folder in root_folder.iterdir() if args.folder_substr in folder.name]
assert len(folders_to_merge) == 7, f"{len(folders_to_merge)}\n{folders_to_merge}"

results_all: Dict[str, Mapping[str, List]] = {"results": defaultdict(list)}
filename = args.filename.split(".json")[0] + ".json"
for i, folder_name in enumerate(folders_to_merge):
    results_file = root_folder / folder_name / filename
    with open(results_file) as f:
        results_current = json.load(f)
    results_all["meta"] = results_current["meta"]
    for frame_name, frame_results in results_current["results"].items():
        for res_dict in frame_results:
            res_dict["tracking_id"] = str(int(res_dict["tracking_id"]) + (MAX_TRACKS_PER_CLASS * i))
        results_all["results"][frame_name].extend(frame_results)

save_folder_name = f"{datetime.datetime.now().strftime('%y-%m-%d')}_merged_results"
save_folder_name += f"{'_' if args.folder_substr[0] != '_' else ''}{args.folder_substr}"
save_folder_name += f"{'_' if args.merged_folder_suffix and args.merged_folder_suffix[0] != '_' else ''}{args.merged_folder_suffix}"

save_folder = root_folder / save_folder_name
makedirs_if_new(str(save_folder))

records_file = save_folder / "params.json"
records = {arg: getattr(args, arg) for arg in vars(args)}
records["folders"] = folders_to_merge
with open(records_file, 'w') as f:
    json.dump(records, f, indent=4)

merged_results_file = save_folder / filename
with open(merged_results_file, 'w') as f:
    json.dump(results_all, f, indent=4)
