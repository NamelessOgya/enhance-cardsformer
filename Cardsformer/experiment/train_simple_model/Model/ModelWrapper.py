from experiment.train_simple_model.Model.PolicyModel import SimplePolicyModel
import torch

device ="0" if torch.cuda.is_available() else "cpu"

class Model:
    """
    The wrapper for the Cardsformer policy model. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, card_dim = 64, bert_dim = 768, embed_dim = 256, dim_ff = 512, device = device, use_text_feature = True):
        """
            device: device_id(str) ex: "0"
        """
        if device != "cpu":
            self.device = f"cuda:{device}"
        else:
            self.device = device
        self.models = {}

        
        print(f"device is {self.device}")
        self.model = SimplePolicyModel(card_dim, bert_dim, embed_dim, dim_ff, use_text_feature = use_text_feature).to(self.device)

    def forward(self, card_embed, minion_embed, secret_embed, weapon_embed, obs, num_options, actor):

        return self.model(
            card_embed.to(self.device), 
            minion_embed.to(self.device), 
            secret_embed.to(self.device), 
            weapon_embed.to(self.device), 
            obs["hand_card_scalar_batch"].to(self.device), 
            obs["minion_scalar_batch"].to(self.device), 
            obs["hero_scalar_batch"].to(self.device), 
            # obs["next_minion_scalar"].to(self.device), 
            # obs["next_hero_scalar"].to(self.device), 
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
