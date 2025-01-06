import os

# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得
print(os.environ["PYTHONNET_RUNTIME"])

from typing import OrderedDict
from Env.Hearthstone import Hearthstone
from Env.EnvWrapper import Environment
from Model.PredictionModel import PredictionModel
from Model.ModelWrapper import Model as PolicyModel

from transformers import AutoModel, AutoTokenizer
import torch
from Algo.encoder import Encoder
import pandas as pd

import logging
import numpy as np
from torch.utils.data import DataLoader

from experiment.util.data_util import EvalPredictionDataset
from experiment.util.model_util import load_prediction_model, load_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_prediction_model(prediction_model_tar_path, eval_data_list):
    """
        prediction_model_tar_path = "./trained_models/prediction_model4715.tar", 
        eval_data_path = ["./off_line_data_vs_policy_model.npy"]
    """
    prediction_model = load_prediction_model(prediction_model_tar_path, is_train=True)
    encoder = load_encoder()
    loss_fn = torch.nn.MSELoss()


    data = EvalPredictionDataset(eval_data_list)
    data_l = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)

    print(len(data_l))
    losses = []
    for n, batch in enumerate(data_l):
        if n % 100 == 0:
            print(f"batch {n} processed")
        
        for i in range(8):
            batch[i] = batch[i].float().to(device)
        x = [batch[i] for i in range(6)] #これはなに
        y = [batch[6], batch[7]] #これはなに
        with torch.no_grad():
            pred = prediction_model(x)

        loss1 = loss_fn(y[0], pred[0])
        loss2 = loss_fn(y[1], pred[1])
        loss = loss1 + loss2
        
        losses.append(loss)
    
    return (sum(losses) / len(losses)).item()


if __name__ == "__main__":
    res = evaluate_prediction_model(
        prediction_model_tar_path = "./trained_models/prediction_model4715.tar", 
        # eval_data_list = ["./off_line_data9.npy"],
        eval_data_list = ["off_line_data_vs_policy_model.npy"],
    )

    print(res)
