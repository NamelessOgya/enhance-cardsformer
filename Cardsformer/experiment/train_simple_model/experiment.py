"""
    python -m experiment.train_simple_model.experiment
"""

import os
import torch
import gc
import wandb
import yaml
from types import SimpleNamespace

from experiment.train_simple_model.train_policy_model import train_policy_model
from experiment.util.experiment_util import get_experiment_code
from experiment.util.wandb_util import generate_experiment_info #api keyと親ディレクトリ名に応じたexperiment_nameを取得
from experiment.train_simple_model.util.model_util import find_best_policy_model



def path_generator(cfg):
    
    return {
        "policy_model_save_path":f"experiment/train_simple_model/res/{cfg.expcode}/policy_models",
        "policy_model_load_path":cfg.policy_model_load_path
    }

info = generate_experiment_info()

if __name__ == "__main__":

    with open(
        "./experiment/train_simple_model/experiment_config/config.yaml", 
        "r", 
        encoding="utf-8"
    ) as f:
        cfg_dict: dict = yaml.safe_load(f) 

    cfg = SimpleNamespace(**cfg_dict)
    cfg.epsilon = float(cfg.epsilon)
    ###################
    ## wandb 関連設定
    ###################

    # wandb
    info = generate_experiment_info()
    wandb.login(key=info["api_key"])
    wandb.init(
        project=cfg.PROJ_NAME,  # プロジェクト名
        name = info["experiment_name"],
        config=cfg_dict
    )
    
    dir_dic = path_generator(cfg)

    # ディレクトリをなければ作る
    for key, value in dir_dic.items():
        if not os.path.exists(value):
            os.makedirs(value)

    print("policy process start")

    # 参照元の重みがあるならloadする。
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
        best_policy_model_dir = best_policy_model_dir,
        deck_mode = cfg.DECK_MODE,
        use_text_feature = cfg.USE_TEXT_FEATURE,
        cfg = cfg
    )

    torch.cuda.empty_cache()
    gc.collect()
    
    wandb.finish()
    
    

