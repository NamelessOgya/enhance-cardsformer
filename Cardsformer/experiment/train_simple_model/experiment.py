"""
    python -m experiment.train_simple_model.experiment
"""

import os
import torch
import gc
import wandb


from experiment.train_simple_model.train_policy_model import train_policy_model
from experiment.util.experiment_util import get_experiment_code
from experiment.util.wandb_util import generate_experiment_info #api keyと親ディレクトリ名に応じたexperiment_nameを取得
from experiment.train_simple_model.util.model_util import find_best_policy_model

def path_generator():
    
    return {
        "policy_model_save_path":f"experiment/train_simple_model/res/{exp_code}/policy_models",
        "policy_model_load_path":"NONE",
    }

TOTAL_POLICY_FRAME = 50000000      #policy modelを学習する際のframe数

PROJ_NAME = "simple_model"
info = generate_experiment_info()
exp_code = info["experiment_name"]


if __name__ == "__main__":
    ###################
    ## wandb 関連設定
    ###################
    

    # wandb
    info = generate_experiment_info()
    wandb.login(key=info["api_key"])
    wandb.init(
        project=PROJ_NAME,  # プロジェクト名
        name = info["experiment_name"],
        config={
            "TOTAL_POLICY_FRAME": TOTAL_POLICY_FRAME
        }
    )
 
    dir_dic = path_generator()

    for key, value in dir_dic.items():
        if not os.path.exists(value):
            os.makedirs(value)

    print("policy process start")

    if dir_dic["policy_model_load_path"] != "NONE":
        try:
            best_policy_model = find_best_policy_model(dir_dic["policy_model_load_path"])
            best_policy_model_dir = dir_dic["policy_model_load_path"] + "/Cardsformer/" + best_policy_model
        except:
            best_policy_model_dir = "NONE"
    else:
        best_policy_model_dir = "NONE"

    train_policy_model(
        model_save_dir = dir_dic["policy_model_save_path"],
        policy_model_load_path =  dir_dic["policy_model_load_path"],
        total_frames = TOTAL_POLICY_FRAME,
        best_policy_model_dir = best_policy_model_dir,
    )

    torch.cuda.empty_cache()
    gc.collect()
    
    wandb.finish()
    
    

