"""
    python -m experiment.prediction_policy_cycle.experiment
"""

import os
import torch
import gc
import wandb

from experiment.prediction_policy_cycle.gen_data import gen_data
from experiment.prediction_policy_cycle.train_prediction_model import train_prediction_model
from experiment.prediction_policy_cycle.train_policy_model import train_policy_model
from experiment.util.model_util import find_best_prediction_model, find_best_policy_model
from experiment.util.experiment_util import get_experiment_code
from experiment.util.wandb_util import generate_experiment_info


def path_generator(cycle):
    
    return {
        "prediction_data_save_path": f"experiment/prediction_policy_cycle/res/{exp_code}/prediction_data/cycle_{cycle + 1}",
        "prediction_data_load_path": f"experiment/prediction_policy_cycle/res/{exp_code}/prediction_data/cycle_{cycle}",

        "prediction_model_save_path": f"experiment/prediction_policy_cycle/res/{exp_code}/prediction_models/cycle_{cycle}",
        
        "policy_model_save_path":f"experiment/prediction_policy_cycle/res/{exp_code}/policy_models/cycle_{cycle}",
        "policy_model_load_path":f"experiment/prediction_policy_cycle/res/{exp_code}/policy_models/cycle_{cycle - 1}" if cycle > 0 else "NONE"
    }

CYCLE = 1
TOTAL_PREDICTION_EPOCH  = 1000       #prediction modelを学習する際のframe数(おおむね)
TOTAL_prediction_FRAMES = 800000  # prediction modelの学習に用いる全フレーム数
TOTAL_POLICY_FRAME = 50000000      #policy modelを学習する際のframe数

CYCLE = 5
TOTAL_PREDICTION_EPOCH  = 5       #prediction modelを学習する際のframe数(おおむね)
TOTAL_prediction_FRAMES = 8000  # prediction modelの学習に用いる全フレーム数
TOTAL_POLICY_FRAME = 50000      #policy modelを学習する際のframe数



POLICY_FRAME_PER_CYCLE = int(TOTAL_POLICY_FRAME/ CYCLE)
TRAIN_DATA_LIMIT  = int(TOTAL_prediction_FRAMES / CYCLE / 10) # 一つのnpyファイルに可能するデータの数


PROJ_NAME = "prediction_policy_cycle"
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
            # "learning_rate": lr,
            # "batch_size": batch_size,
            "CYCLE": CYCLE,
            "TOTAL_PREDICTION_EPOCH":TOTAL_PREDICTION_EPOCH,
            "TOTAL_prediction_FRAMES":TOTAL_prediction_FRAMES,
            "TOTAL_POLICY_FRAME": TOTAL_POLICY_FRAME,
            "POLICY_FRAME_PER_CYCLE": POLICY_FRAME_PER_CYCLE,
            "TRAIN_DATA_LIMIT":TRAIN_DATA_LIMIT
        }
    )

    ###########################


    for i in range(CYCLE):
        print(f"=== iter {i} start ===")

        
        dir_dic = path_generator(i)

        for key, value in dir_dic.items():
            if not os.path.exists(value):
                os.makedirs(value)
    
        if i == 0:
            # 初回限定でgen_dataを実行
            print("==== gen data start ====")
            gen_data(dir_dic["prediction_data_load_path"], TRAIN_DATA_LIMIT)

        print("==== prediction start ====")
        train_prediction_model(
            data_dir  = dir_dic["prediction_data_load_path"],
            model_dir = dir_dic["prediction_model_save_path"],
            best_model_path = "NONE" if i == 0 else best_model_dir,
            train_step = TOTAL_PREDICTION_EPOCH

        )

        best_model = find_best_prediction_model(dir_dic["prediction_model_save_path"])
        best_model_dir = dir_dic["prediction_model_save_path"] + "/" + best_model

        print("policy process start")

        if dir_dic["policy_model_load_path"] != "NONE":
            best_policy_model = find_best_policy_model(dir_dic["policy_model_load_path"])
            best_policy_model_dir = dir_dic["policy_model_load_path"] + "/Cardsformer/" + best_policy_model
        else:
            best_policy_model_dir = "NONE"

        train_policy_model(
            model_save_dir = dir_dic["policy_model_save_path"],
            prediction_model = best_model_dir,
            prediction_data_save_path = dir_dic["prediction_data_save_path"],
            policy_model_load_path =  dir_dic["policy_model_load_path"],
            best_policy_model_dir = best_policy_model_dir,
            total_frames = POLICY_FRAME_PER_CYCLE * (i + 1),
            train_data_limit = TRAIN_DATA_LIMIT
        )

        torch.cuda.empty_cache()
        gc.collect()
    
    wandb.finish()
    
    

