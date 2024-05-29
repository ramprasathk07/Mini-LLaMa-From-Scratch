import torch
import torch.nn as nn
import torch.functional as F
from dataset import MASTER_CONFIG

#-------------------------------------------------------------------------------------
# Definition of a basic neural network class
class LLama(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config
        # Embedding layer to convert character indices to vectors (vocab size: 65)
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),  # Currently using ReLU, will be replaced with SwiGLU as in LLaMA
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # Print the total number of model parameters
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self,idx,targets = None):
        x = self.embedding(idx)
        a = self.linear(x)

        logits = F.softmax(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits,loss

        else:
            return logits