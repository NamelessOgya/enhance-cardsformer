import numpy as np
from torch.utils.data import Dataset

class NpyLogData:
    def __init__(self):
        self.data = {
            'hand_card_names': [],
            'minion_names': [],
            'weapon_names': [],
            'hand_card_scalar': [],
            'minion_scalar': [],
            'hero_scalar': [],
            'next_state_minion_scalar': [],
            'next_state_hero_scalar': [],
        }

    def save_data(self, old_obs, action_idx,next_state, encoder):
        self.data['hand_card_names'].append(encoder.encode(old_obs['hand_card_names']).detach().cpu().numpy())
        self.data['minion_names'].append(encoder.encode(old_obs['minion_names']).detach().cpu().numpy())
        self.data['weapon_names'].append(encoder.encode(old_obs['weapon_names']).detach().cpu().numpy())
        self.data['hand_card_scalar'].append(old_obs['hand_card_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
        self.data['minion_scalar'].append(old_obs['minion_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
        self.data['hero_scalar'].append(old_obs['hero_scalar_batch'][action_idx].detach().cpu().numpy().astype(np.int64))
        self.data['next_state_minion_scalar'].append(next_state['minion_scalar'].detach().cpu().numpy().astype(np.int64))
        self.data['next_state_hero_scalar'].append(next_state['hero_scalar'].detach().cpu().numpy().astype(np.int64))

    def flatten_batch_np_array(self, ar):
        if len(ar.shape) == 4:
            return ar.reshape([-1, ar.shape[2], ar.shape[3]])
        else:
            return ar.reshape([-1, ar.shape[2]])

    def save_data_from_buffer(self, batch):
        hcn_npy  = self.flatten_batch_np_array(batch['hand_card_embed'].detach().cpu().numpy())
        mn_npy   = self.flatten_batch_np_array(batch['minion_embed'].detach().cpu().numpy())
        we_npy   = self.flatten_batch_np_array(batch['weapon_embed'].detach().cpu().numpy())
        hcs_npy  = self.flatten_batch_np_array(batch['hand_card_scalar'].detach().cpu().numpy().astype(np.int64))
        ms_npy   = self.flatten_batch_np_array(batch['minion_scalar'].detach().cpu().numpy().astype(np.int64))
        hs_npy   = self.flatten_batch_np_array(batch['hero_scalar'].detach().cpu().numpy().astype(np.int64))
        nsms_npy = self.flatten_batch_np_array(batch['next_minion_scalar'].detach().cpu().numpy().astype(np.int64))
        nshs_npy = self.flatten_batch_np_array(batch['next_hero_scalar'].detach().cpu().numpy().astype(np.int64))

        self.data['hand_card_names'].extend([hcn_npy[i] for i in range(len(hcn_npy))])
        self.data['minion_names'].extend([mn_npy[i] for i in range(len(mn_npy))])
        self.data['weapon_names'].extend([we_npy[i] for i in range(len(we_npy))])
        self.data['hand_card_scalar'].extend([hcs_npy[i] for i in range(len(hcs_npy))])
        self.data['minion_scalar'].extend([ms_npy[i] for i in range(len(ms_npy))])
        self.data['hero_scalar'].extend([hs_npy[i] for i in range(len(hs_npy))])
        self.data['next_state_minion_scalar'].extend([nsms_npy[i] for i in range(len(nsms_npy))])
        self.data['next_state_hero_scalar'].extend([nshs_npy[i] for i in range(len(nshs_npy))])

    def save_to_npy(self, npy_name='off_line_data_vs_ai.npy'):
        np.save(npy_name, self.data)

    def __len__(self):
        # 全てのデータのキーでリストの長さが同じと仮定して1つをチェック
        return len(self.data['hand_card_names'])

class EvalPredictionDataset(Dataset):

    def __init__(self, data_list):
        """
            ["./off_line_data_vs_policy_model.npy"]
        """
        super(Dataset, self).__init__()
        self.data = {
            'hand_card_names': [],
            'minion_names': [],
            'weapon_names': [],
            'hand_card_scalar': [],
            'minion_scalar': [],
            'hero_scalar': [],
            'next_state_minion_scalar': [],
            'next_state_hero_scalar': [],

        }
        for data_path in data_list:
            print(f"=== load prediction dataset:{data_path} ===")
            
            cur_data = np.load(data_path, allow_pickle=True).item()
            for key in cur_data:
                if key == 'secret_names':
                    continue
                self.data[key] += cur_data[key]

    def __len__(self):
        return len(self.data['hand_card_names'])

    def __getitem__(self, index, id=False):
        return self.data['hand_card_names'][index], self.data['minion_names'][index], self.data['weapon_names'][index], self.data['hand_card_scalar'][index], self.data['minion_scalar'][index], self.data['hero_scalar'][index], self.data['next_state_minion_scalar'][index], self.data['next_state_hero_scalar'][index]
    