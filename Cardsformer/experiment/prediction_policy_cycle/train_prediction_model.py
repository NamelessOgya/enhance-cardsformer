from typing import OrderedDict
import torch
from torch.utils.data import DataLoader
from experiment.prediction_policy_cycle.PredictionDataset import PredictionDataset # experiment専用のprediction dataset
from tqdm import tqdm
from Model.PredictionModel import PredictionModel
from experiment.util.model_util import load_prediction_model
import torch.optim
from Algo.utils import log
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import os

import configparser
import wandb
from datetime import datetime, timezone, timedelta

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

##### configs ########
mode = "test"
batch_size = 512
lr = 0.0001
######################



# モデルをウォッチ（オプション：勾配などを追跡）
best_test_loss = float('inf')

#######################################


def train_prediction_model(data_dir, model_dir, best_model_path, train_step):
    """
        data_dir: experiment/prediction_policy_cycle/prediction_data/cycle_0
        model_dir: /experiment/prediction_policy_cycle/prediction_models/cycle_0
        prediction_model_load
    """
    model = PredictionModel(is_train=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # wandb.watch(model, log="all")

    ## 前のサイクルで学習済みのモデルがあるなら復元
    if best_model_path != "NONE":
        checkpoint = torch.load(best_model_path)
        model = load_prediction_model(best_model_path, is_train=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    data = PredictionDataset([i for i in range(9)] if not mode == "debug" else [1], data_dir=data_dir)
    data_test = PredictionDataset([9], data_dir=data_dir) 
    # test dataないのでtrain dataの一番最後をtestに。
    # TODO test dataを追加（もしくは、学習データをもう一ファイル増やす。）

    data_train = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
    data_test = DataLoader(data_test, batch_size=5000, shuffle=True)


    device = 'cpu' ## <-?????
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = torch.nn.DataParallel(model).to(f"cuda:{device}")
    loss_fn = torch.nn.MSELoss()
    
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
        wandb.log({"PREDICTION_training_loss": np.mean(losses), "epoch": epoch}) # add
        log.info("Current Training Loss is: {}".format(loss))
        test_losses = []
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
        test_loss = np.mean(test_losses)
        wandb.log({"PREDICTION_test_loss": test_loss, "epoch": epoch})
        writer.add_scalar('test_loss', test_loss, epoch)
        if test_loss < best_test_loss:
            log.info('minion loss: {} \t hero loss: {}'.format(loss1, loss2))
            log.info('Best loss: {}'.format(test_loss))
            
            best_test_loss = test_loss
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },  model_dir + '/prediction_model' + str(epoch) + '.tar'
                )

            wandb.log({"PREDICTION_best_test_loss": best_test_loss, "model_path": 'trained_models/prediction_model' + str(epoch) + '.tar'})
            
        writer.add_scalar('best_test_loss', best_test_loss, epoch)
        wandb.log({"PREDICTION_best_test_loss": best_test_loss, "epoch": epoch})

    