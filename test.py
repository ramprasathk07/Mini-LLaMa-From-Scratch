import torch 
import numpy as np 
from dataset import get_batches

@torch.no_grad
def evaluate_loss(model):
    out = {}
    model.eval()

    for split in ['train','val']:
        losses = []
        for _ in range(20):
            xb,yb = get_batches(split=split)

            _,loss = model(xb,yb)

            losses.append(loss.item())
    
    out[split] = np.mean(losses)
    
    return out