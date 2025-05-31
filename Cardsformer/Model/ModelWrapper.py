from Model.PolicyModel import PolicyModel
import torch
import re

device ="cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

def normalize_device(device: str) -> str:
    """
    device が
      - 数字のみの文字列 ("0", "1", ...)
        → "cuda:{数字}" を返す
      - "cuda:数字" の形式
        → そのまま返す
      - それ以外
        → ValueError を投げる
    """
    if device == "cpu":
        return "cpu"
    if not isinstance(device, str):
        raise TypeError(f"device must be str, but got {type(device).__name__}")

    # 数字のみ ("0", "1", "10" など)
    if device.isdigit():
        return f"cuda:{device}"

    # すでに "cuda:数字" 形式
    if re.fullmatch(r"cuda:\d+", device):
        return device

    # それ以外はエラー
    raise ValueError(f"invalid device format: {device!r}")

class Model:
    """
    The wrapper for the Cardsformer policy model. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, card_dim = 64, bert_dim = 768, embed_dim = 256, dim_ff = 512, device = device):
        """
            device: device_id(str) ex: "0"
        """
        self.device = normalize_device(device)
        self.models = {}
        
        print(f"device is {self.device}")
        self.model = PolicyModel(card_dim, bert_dim, embed_dim, dim_ff).to(self.device)

    def forward(self, card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor):

        return self.model(
            card_embed.to(self.device), 
            minion_embed.to(self.device), 
            secret_embed.to(self.device), 
            weapon_embed.to(self.device), 
            obs["hand_card_scalar_batch"].to(self.device), 
            obs["minion_scalar_batch"].to(self.device), 
            obs["hero_scalar_batch"].to(self.device), 
            obs["next_minion_scalar"].to(self.device), 
            obs["next_hero_scalar"].to(self.device), 
            num_options, 
            actor
        )

    def share_memory(self):
        self.model.share_memory()
        return

    def eval(self):
        self.model.eval()
        return

    def parameters(self):
        return self.model.parameters()

    def get_model(self):
        return self.model
