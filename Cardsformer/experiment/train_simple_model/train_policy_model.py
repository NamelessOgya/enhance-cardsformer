"""
    - policy modelを学習しながらprediction modelのサンプルを用意する。
	 ⇒ play_dataの保存ができるような工夫が必要

	python -m experiment.train_simple_model.train_policy_model
"""

from Algo.arguments import parser
import os
import signal
# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得
print(os.environ["PYTHONNET_RUNTIME"])
import threading
import time
import timeit  # 测试代码片段执行时间
import pprint  # pretty print 一种美化输出的工具
from collections import deque  # 双向队列
import numpy as np

import torch
# import multiprocessing as mp
from torch import multiprocessing as mp  # torch提供的多线程，如何使用？
from torch import nn


from experiment.train_simple_model.Model.ModelWrapper import Model  # 网络模型
# from Algo.utils import get_batch, log, create_buffers, create_optimizers, act  # 这几个函数是干什么用的？
from experiment.train_simple_model.act.act_util import get_batch, log, create_buffers, create_optimizers, act

import wandb


from experiment.util.data_util import NpyLogData
from experiment.train_simple_model.util.battle_util import evaluate_model_with_rulebase

import gc


mean_episode_return_buf = {
    p: deque(maxlen=100)
    for p in ['Player1', 'Player2']
}

def compute_loss(logits, targets):
    loss = ((logits.view(-1) - targets)**2).mean()
    return loss

def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    hand_card_embed = batch['hand_card_embed'].to(device)
    minion_embed = batch['minion_embed'].to(device)
    weapon_embed = batch['weapon_embed'].to(device)
    secret_embed = batch['secret_embed'].to(device)
    hand_card_scalar = batch['hand_card_scalar'].to(device)
    minion_scalar = batch['minion_scalar'].to(device)
    hero_scalar = batch['hero_scalar'].to(device)
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(
        torch.mean(episode_returns).to(device))

    next_minion_scalar = batch['next_minion_scalar'].to(device)
    next_hero_scalar = batch['next_hero_scalar'].to(device)
    with lock:
        learner_outputs = model(
            hand_card_embed, 
            minion_embed, 
            secret_embed, 
            weapon_embed, 
            hand_card_scalar, 
            minion_scalar, 
            hero_scalar, 
            num_options=None, 
            actor = False
        )
        loss = compute_loss(learner_outputs, target)
        stats = {
            'mean_episode_return_' + position:
            torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]
                             ])).item(),
            'loss_' + position:
            loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model().load_state_dict(model.state_dict())
        return stats

def train(
        flags,
        policy_model_load_path,
        best_policy_model_dir,
        deck_mode = None,
        use_text_feature = True
    ):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
	##### npyデータ保存用にデータセット作成 #####
    npy_data = NpyLogData()
    num = 1

    ##### 結果格納dirの作成 ###########
    model_save_dir = os.path.expanduser('%s/%s' %
                            (flags.savedir, flags.xpid))
        
    print(f"save_model_dir : {model_save_dir}")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    ##################################
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`"
            )
    checkpointpath = os.path.expanduser('%s/%s/%s' %
                           (flags.savedir, flags.xpid, 'model.tar'))
    
    
    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['1']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')
        ), 'The number of actor devices can not exceed the number of available devices'

    models = {}
    for device in device_iterator:
        model = Model(device=str(device), use_text_feature = use_text_feature)
        model.share_memory()
        model.eval()
        models[device] = model
    buffers = create_buffers(flags, device_iterator)

    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
    for device in device_iterator:
        _free_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        _full_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue
    learner_model = Model(device=flags.training_device,use_text_feature = use_text_feature)


    optimizer = create_optimizers(flags, learner_model)
    stat_keys = [
        'mean_episode_return_Player1',
        'loss_Player1',
        'mean_episode_return_Player2',
        'loss_Player2',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'Player1': 0, 'Player2': 0}

    # Load models if any

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(policy_model_load_path)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    if policy_model_load_path != "NONE":
        checkpoint_states = torch.load(
            policy_model_load_path + "/Cardsformer/model.tar",
            map_location=("cuda:" + str(flags.training_device)
                          if flags.training_device != "cpu" else "cpu"))
        learner_model.get_model().load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        for device in device_iterator:
            models[device].get_model().load_state_dict(learner_model.get_model().state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    actor_processes = []
    parent_stop_event, child_stop_event = mp.Pipe()
    
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(num_actors):
            actor = ctx.Process(target=act,
                                args=(i, device, free_queue[device],
                                      full_queue[device], models[device],
                                      buffers[device], flags, child_stop_event))
            actor.start()
            actor_processes.append(actor)

     # ─── ここで親プロセス側の「子受信用端点」を閉じる ───  ← 修正
    child_stop_event.close() 

    def batch_and_learn(i,
                        device,
                        position,
                        local_lock,
                        position_lock,
                        lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position],
                              full_queue[device][position],
                              buffers[device][position], flags, local_lock)
            if frames % 10000 == 0:
                # バッチを保存
                npy_data.save_data_from_buffer(batch)
                # 保存後に内部バッファ（self.data）をクリア
                for k in npy_data.data:
                    npy_data.data[k].clear()
            _stats = learn(position, models, learner_model.get_model(), batch, optimizer, flags, position_lock)
            
             # バッチ自体も消す
            del batch
            gc.collect()

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['Player1'].put(m)
            free_queue[device]['Player2'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {
            'Player1': threading.Lock(),
            'Player2': threading.Lock()
        }
    position_locks = {
        'Player1': threading.Lock(),
        'Player2': threading.Lock()
    }

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['Player1', 'Player2']:
                thread = threading.Thread(target=batch_and_learn,
                                          name='batch-and-learn-%d' % i,
                                          args=(i, device, position,
                                                locks[device][position],
                                                position_locks['Player1']))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _model = learner_model.get_model()
        torch.save(
            {
                'model_state_dict': _model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "stats": stats,
                'flags': vars(flags),
                'frames': frames,
                'position_frames': position_frames
            }, checkpointpath)

        # Save the weights for evaluation purpose
        model_weights_dir = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' %
                                (flags.savedir, flags.xpid, 'Trained_weights_' + str(frames) + '.ckpt')))
        torch.save(
            learner_model.get_model().state_dict(),
            model_weights_dir)
        
        res_dic = {}
        for i in ["RandomAgent", "GreedyAgent"]:
            print(f"=== evaluating vs {i}... ===")
            win_rate = evaluate_model_with_rulebase(
                check_model_dir = model_weights_dir,
                rule_model_name = i,
                match_num = 100,
                device = "cpu",
                deck_mode = deck_mode,
                use_text_feature = use_text_feature
            )
            print(f"evaluation done {win_rate}")
            res_dic[f"WIN_RATE_against_{i}"] = win_rate


            

        wandb.log(res_dic)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_save_frame = frames - (frames % flags.frame_interval)
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {
                k: position_frames[k]
                for k in position_frames
            }
            start_time = timer()
            time.sleep(5)

            if frames - last_save_frame > flags.frame_interval:
                print("mdoel saved!")
                checkpoint(frames - (frames % flags.frame_interval))
                last_save_frame = frames - (frames % flags.frame_interval)
            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {
                k: (position_frames[k] - position_start_frames[k]) /
                (end_time - start_time)
                for k in position_frames
            }
            log.info(
                'After %i (L:%i U:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f) Stats:\n%s',
                frames,
                position_frames['Player1'],
                position_frames['Player2'], fps, fps_avg,
                position_fps['Player1'], position_fps['Player2'], pprint.pformat(stats)
            )

            wandb.log({
                "POLOCY_frames": frames,
                "POLOCY_Player1_frames": position_frames['Player1'],
                "POLOCY_Player2_frames": position_frames['Player2'],
                "POLOCY_fps": fps,
                "POLOCY_fps_avg": fps_avg,
                "POLOCY_Player1_fps": position_fps['Player1'],
                "POLOCY_Player2_fps": position_fps['Player2'],
                "POLOCY_stats": stats  # statsはそのまま辞書形式で記録
            })

			
            # npyデータのセーブは一旦コメントアウト
            # if len(npy_data) > train_data_limit:
            #     npy_data.save_to_npy(prediction_data_save_path + f"/off_line_data{(num-1) % 10}.npy")
            #     npy_data = NpyLogData()
            #     num += 1

        print("data_saved!!!")
        # parent_stop_event.send("STOP")
        # for thread in threads:
        #     thread.join()
        # print("thread joined!!!")

         # メインの学習ループが終わったら
        parent_stop_event.send("STOP")
        # ─── メッセージ送信直後に送信端点を閉じる ───  ← 修正
        parent_stop_event.close()
        for actor in actor_processes:
            actor.join()
            actor.terminate()
            print("process terminated")
        torch.cuda.empty_cache()
        
        child_stop_event.close() 
        gc.collect()

    except KeyboardInterrupt:
        return
    
    print("every thing is done")
    # else:
    #     for thread in threads:
    #         thread.join()
    #     log.info('Learning finished after %d frames.', frames)


def train_policy_model(
        model_save_dir,
        policy_model_load_path,
        best_policy_model_dir,
        total_frames,
        deck_mode = None,
        use_text_feature = True

    ):
    flags = parser.parse_args()
    flags.total_frames = total_frames
    flags.frame_interval = 100000 # デバッグ用暫定対処
    flags.gpu_devices = "1"
    flags.num_actors = 2
    print("==================")
    print(flags)
    print("==================")
	
    print(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    flags.savedir = model_save_dir
    print(f"== save_dir = {flags.savedir} ==")
    
    print(f"total frames ======== {flags.total_frames}")
    train(
        flags,
        policy_model_load_path,
        best_policy_model_dir,
        deck_mode,
        use_text_feature = use_text_feature
    )

