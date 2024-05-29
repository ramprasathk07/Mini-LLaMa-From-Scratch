import torch 
from model import LLama
from data import MASTER_CONFIG,get_batches
import time 

def train(model, optimizer, 
          scheduler=None, 
          config=MASTER_CONFIG, 
          print_logs=False):
    
    losses = []

    start_time = time.time()

    for epoch in range(config['epochs']):

        optimizer.zero_grad()
        xs,ys = get_batches(train,)
