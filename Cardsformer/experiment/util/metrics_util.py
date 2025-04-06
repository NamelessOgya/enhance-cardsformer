import torch

def accuracy_per_item(pred, label): 
    pred_int = torch.floor(pred) #(batch, entity, item)
    label_int = torch.floor(label) #(batch, entity, item)

    res = pred_int == label_int #(batch, entity, item)

    res = res.sum(dim=(0,1)) / (pred_int.shape[0] * pred_int.shape[1])  #(item)

    return res #(item)
