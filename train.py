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

def train(model, optimizer,
          scheduler=None, 
          config=MASTER_CONFIG,
          ):
    
    losses = []

    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
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

            y = {'train':loss.item()}
            losses +=[y]

            x = evaluate_loss(model)
            losses +=[x]
            print(f"Epoch {epoch} | Train loss {y['train']:.3f} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
            logging.info(f"Epoch {epoch} | Train loss {y['train']:.3f} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
            start_time = time.time()

    train_losses = [d['train'] for d in losses if 'train' in d]
    val_losses = [d['val'] for d in losses if 'val' in d]
    pd.DataFrame({'train': train_losses, 'val': val_losses}).plot()
    plt.show()

    return model
    
if __name__ == '__main__':
    log = 'Logs'
    os.makedirs(log,exist_ok=True)
    logging.basicConfig(filename=f'{log}/Train.log', filemode='a', format='%(asctime)s - %(message)s',level=logging.INFO)
    logging.info('Training started.')
    model = LLama()
    optimizer = AdamW(model.parameters())
    model = train(model, optimizer)
    e = MASTER_CONFIG['epochs']
    path = MASTER_CONFIG['checkpoint']
    os.makedirs(path,exist_ok=True)
    logging.info('Training completed.')
    torch.save(model.state_dict(), f'{path}/model_state_dict_{e}.pth')
    torch.save(model, f'{path}/model_{e}.pt')

        