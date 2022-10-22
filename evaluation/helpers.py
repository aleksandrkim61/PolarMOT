import time
from pathlib import Path
from typing import Iterable, List, Any

import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report

from models.graph_tracker_offline import GraphTrackerOffline


def predict_for(ckpt_path, dataloader, print_inference_time: bool = False):
    ckpt_path = str(ckpt_path)
    print(f"Loading {ckpt_path}")
    hparams_file = ckpt_path.split("checkpoints")[0] + "hparams.yaml"
    model_to_test = GraphTrackerOffline.load_from_checkpoint(ckpt_path, hparams_file=hparams_file)
    start_time = time.time()
    res = pl.Trainer(gpus=1).predict(model_to_test, dataloaders=dataloader)
    if print_inference_time:
        print(f"Inference took {time.time() - start_time:.1f} sec")
    return res


def classification_report_for(ckpt_path, dataloader, *, output_dict: bool = False, print_inference_time: bool = False, threshold:float = 0.5):
    preds_ys = predict_for(ckpt_path, dataloader, print_inference_time)

    preds_list, ys_list = zip(*preds_ys)
    preds = torch.cat(preds_list)
    ys = torch.cat(ys_list)
    if threshold == 0.5:
        preds = preds.numpy().round().astype(int)
    else:
        preds = (preds > threshold).int()

    class_report = classification_report(y_true=ys.int(), y_pred=preds,
                                         output_dict=output_dict, zero_division=0)
    if not output_dict:
        print(class_report)
    else:
        return class_report
