import torch 
torch.autograd.set_detect_anomaly(True)
from torch.optim import AdamW
from model import LLama
import time 
import logging 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from dataset import MASTER_CONFIG,get_batches
from eval import evaluate_loss
log = 'Logs'
os.makedirs(log,exist_ok=True)

logging.basicConfig(filename=f'{log}/train.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')

def train(model, optimizer, 
          scheduler=None, 
          config=MASTER_CONFIG,):
    
    losses = []

    start_time = time.time()

    for epoch in range(config['epochs']):

        optimizer.zero_grad()
        xs,ys = get_batches(split = train,
                            batch_size = MASTER_CONFIG['batch_size'], 
                            context_window = MASTER_CONFIG['batch_size'],
                            )
        
        logits,loss = model(xs,targets = ys)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
            print("lr: ", scheduler.get_lr())


        if epoch % MASTER_CONFIG['log_interval'] == 0:
            batch_time = time.time()-start_time

            x = evaluate_loss(model)
            losses +=[x]
            print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")

            start_time = time.time()
        
    print("Validation loss: ", losses[-1]['val'])
    pd.DataFrame(losses).plot()
    plt.show()
    return 
    
model = LLama()
optimizer = AdamW(model.parameters())
train(model, optimizer)

        