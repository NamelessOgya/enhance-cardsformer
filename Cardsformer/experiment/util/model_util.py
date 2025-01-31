"""
    python -m experiment.util.model_util
"""

import torch
from typing import OrderedDict
from transformers import AutoModel, AutoTokenizer

from Algo.encoder import Encoder
from Model.ModelWrapper import Model as PolicyModel
from Model.PredictionModel import PredictionModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_prediction_model(prediction_model_tar, is_train=False):
    """
        prediction_model_tar
            path of prediction_model
                "./trained_models/prediction_model4715.tar"
    """
    if is_train:
        prediction_model = PredictionModel(is_train=True)
    else:    
        prediction_model = PredictionModel()
    # prediction
    checkpoint_states = torch.load(prediction_model_tar, map_location=device)['model_state_dict']

    # unwrap the prediction model
    new_state_dict = OrderedDict()
    for k, v in checkpoint_states.items():
        name = k[7:]
        new_state_dict[name] = v

    prediction_model.load_state_dict(new_state_dict)
    prediction_model.to(device)
    prediction_model.eval()

    return prediction_model

def load_policy_model(checkpoint_path):
    model =PolicyModel(device=device)
    checkpoint_states = torch.load(checkpoint_path,map_location=device)
    model.get_model().load_state_dict(checkpoint_states)  

    return model


def load_encoder():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    auto_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    encoder = Encoder(model=auto_model, tokenizer=tokenizer)
    encoder.to(device)

    return encoder

def find_best_prediction_model(model_dir):
    """
        model_dir: prediction_model0.tar 形式のモデルが入ったディレクトリ
    """
    max_id = 0
    max_file_name = "NOT_FOUND"
    for filename in os.listdir(model_dir):
        file_id = int(filename.split("prediction_model")[1].split(".tar")[0])
        if file_id >= max_id:
            max_id = file_id
            max_file_name = filename
    return max_file_name

def find_best_policy_model(model_dir):
    model_dir = model_dir + "/Cardsformer"
    max_id = 0
    max_file_name = "NOT_FOUND"
    for filename in os.listdir(model_dir):
        if "Trained_weights_" in filename:
            file_id = int(filename.split("Trained_weights_")[1].split(".ckpt")[0])
            if file_id > max_id:
                max_id = file_id
                max_file_name = filename
    return max_file_name


if __name__ == "__main__":
    print(find_best_prediction_model("./experiment/prediction_policy_cycle/prediction_models/cycle_0"))