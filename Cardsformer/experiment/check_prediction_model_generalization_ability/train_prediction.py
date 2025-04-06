# python -m experiment.check_prediction_model_generalization_ability.train_prediction
import os

# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得Z
print(os.environ["PYTHONNET_RUNTIME"])


from typing import OrderedDict
import torch
from torch.utils.data import DataLoader
from PredictionDataset import PredictionDataset
from tqdm import tqdm
from Model.PredictionModel import PredictionModel
import torch.optim
from Algo.utils import log
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import os

import configparser
import wandb
from datetime import datetime, timezone, timedelta

from experiment.util.metrics_util import accuracy_per_item

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

##### configs ########
mode = "test"
batch_size = 8000
lr = 0.0001
train_step = 5000
######################

### add wandb logger ##################
# get current_dir_name
# 現在のディレクトリのパスを取得
current_dir = os.getcwd()

# 一つ上のディレクトリのパスを取得
parent_dir = os.path.dirname(current_dir)

# 一つ上のディレクトリ名を取得
parent_dir_name = os.path.basename(parent_dir)

PROJ_NAME = "reproduce-cards-former-debug" if mode == "debug" else "check-prediction-accuracy"
COMMON_CONFIG_PATH = os.path.abspath("../config/config.ini")

config_ini = configparser.ConfigParser()
config_ini.read(COMMON_CONFIG_PATH, encoding='utf-8')
api_key = config_ini['WANDB']['api_key']

# WandBの初期化
# JSTの現在時刻を取得
jst = timezone(timedelta(hours=9))  # JSTのタイムゾーン
current_time = datetime.now(jst)
experiment_name = current_time.strftime("EXP_%Y%m%d_%H%M")  # フォーマット: EXP_[年][月][日]_[時間][分]

wandb.login(key=api_key)
wandb.init(
    project=PROJ_NAME,  # プロジェクト名
    name = parent_dir_name + "_" + experiment_name,
    config={
        "learning_rate": lr,
        "batch_size": batch_size,
        # "optimizer": optimizer.__class__.__name__,
        # "loss_function": loss_fn.__class__.__name__,
    }
)

# モデルをウォッチ（オプション：勾配などを追跡）
best_test_loss = float('inf')

#######################################

model = PredictionModel(is_train=True)
wandb.watch(model, log="all")



data = PredictionDataset(
    [i for i in range(10)] if not mode == "debug" else [1],
    base_path="./experiment/check_prediction_model_generalization_ability/off_line_data",
    test=False
)
data_test = PredictionDataset(
    [0], 
    base_path="./experiment/check_prediction_model_generalization_ability/off_line_data",
    test=True
) 

# test dataないのでtrain dataの一番最後をtestに。
# TODO test dataを追加（もしくは、学習データをもう一ファイル増やす。）

data_train = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
data_test = DataLoader(data_test, batch_size=5000, shuffle=True)


device = 'cpu' ## <-?????
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
best_test_loss = 100

writer = SummaryWriter('runs/prediction_model')

for epoch in range(train_step):
    log.info("Epoch {} / {}".format(epoch, train_step))
    losses = []
    for batch in data_train:
        optimizer.zero_grad()
        for i in range(8):
            batch[i] = batch[i].float().to(device)
        x = [batch[i] for i in range(6)]
        y = [batch[6], batch[7]]
        pred = model(x)
        loss1 = loss_fn(y[0], pred[0])
        loss2 = loss_fn(y[1], pred[1])
        loss = loss1 + loss2
        losses.append(loss.mean().item())
        loss.backward()
        optimizer.step()
    writer.add_scalar('training_loss', np.mean(losses), epoch)
    wandb.log({"training_loss": np.mean(losses), "epoch": epoch}) # add
    log.info("Current Training Loss is: {}".format(loss))
    test_losses = []
    test_ac_minion = []
    test_ac_hero = []

    for batch in data_test:
        for i in range(3, 8):
            batch[i] = batch[i].float().to(device)
        x = [batch[i] for i in range(6)] 
        y = [batch[6], batch[7]]
        with torch.no_grad():
            pred = model(x)
        loss1 = loss_fn(y[0], pred[0])
        loss2 = loss_fn(y[1], pred[1])
        loss = loss1 + loss2
        test_losses.append(loss.mean().item())
        
        test_ac_minion.append(accuracy_per_item(y[0], pred[0]))
        test_ac_hero.append(accuracy_per_item(y[1], pred[1]))
        


    test_loss = np.mean(test_losses)
    test_ac_hero = torch.stack(test_ac_hero).mean(dim=0).to("cpu").numpy()
    test_ac_minion = torch.stack(test_ac_minion).mean(dim=0).to("cpu").numpy()

    wandb.log({"test_loss": test_loss, "epoch": epoch})
    writer.add_scalar('test_loss', test_loss, epoch)

    for n, i in enumerate(test_ac_hero):
        wandb.log({f"hero_ac_dim{n}": i, "epoch": epoch})
    
    for n, i in enumerate(test_ac_minion):
        wandb.log({f"minion_ac_dim{n}": i, "epoch": epoch})
    

    if test_loss < best_test_loss:
        log.info('minion loss: {} \t hero loss: {}'.format(loss1, loss2))
        log.info('Best loss: {}'.format(test_loss))
        
        best_test_loss = test_loss
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './experiment/check_prediction_model_generalization_ability/trained_models/prediction_model' + str(epoch) + '.tar'
            )

        wandb.log({"best_test_loss": best_test_loss, "model_path": './experiment/check_prediction_model_generalization_ability/trained_models/prediction_model' + str(epoch) + '.tar'})
        
    writer.add_scalar('best_test_loss', best_test_loss.item(), epoch)
    wandb.log({"best_test_loss": best_test_loss, "epoch": epoch})

wandb.finish()