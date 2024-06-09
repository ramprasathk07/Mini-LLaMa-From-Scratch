import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset import MASTER_CONFIG
from utils.llama_feats import RMSNorm,RoPEMaskedMultiheadAttention
import torch
torch.autograd.set_detect_anomaly(True)

#-------------------------------------------------------------------------------------
# Definition of a basic neural network class

class LLama(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config
        # Embedding layer to convert character indices to vectors (vocab size: 65)
        self.embedding = nn.Embedding(int(config['vocab_size']), config['d_model'])
        self.RMSnorm = RMSNorm((config['batch_size'], config['d_model']))

        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )
        
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        # Print the total number of model parameters
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self,idx,targets = None):
        x = self.embedding(idx)

        x = self.RMSnorm(x)
        x += self.rope_attention(x)
        x = self.RMSnorm(x)
        x += self.linear(x)

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits,loss

        else:
            return logits