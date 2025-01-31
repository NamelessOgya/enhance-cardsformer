import os
import configparser
from datetime import datetime, timezone, timedelta

def generate_experiment_info():
    COMMON_CONFIG_PATH = os.path.abspath("../config/config.ini")

    config_ini = configparser.ConfigParser()
    config_ini.read(COMMON_CONFIG_PATH, encoding='utf-8')
    api_key = config_ini['WANDB']['api_key']


    # 一つ上のディレクトリのパスを取得
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    # 一つ上のディレクトリ名を取得
    parent_dir_name = os.path.basename(parent_dir)

    jst = timezone(timedelta(hours=9))  # JSTのタイムゾーン
    current_time = datetime.now(jst)
    experiment_name = current_time.strftime("EXP_%Y%m%d_%H%M")  # フォーマット: EXP_[年][月][日]_[時間][分]

    return {
        "api_key": api_key,
        "experiment_name": parent_dir_name + "_" + experiment_name
    }