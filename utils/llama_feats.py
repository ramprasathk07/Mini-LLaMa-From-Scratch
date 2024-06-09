import torch 
import torch.nn as nn
import numpy as np
from dataset import MASTER_CONFIG as config
from torch.nn import functional as F

class RMSNorm(nn.Module):

    '''
    Root Mean Square Normalization

    '''
    def __init__(self,
                 layer_shape,
                 eps = 1e-8,
                 bias = False):
        super(RMSNorm,self).__init__()

        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self,x):
        #shape -> batch,seq_len,d_model

        rms = torch.linalg.norm(x,dim = (1,2))*x[0].numel()**(-0.5)

        raw = x / rms.unsqueeze(-1).unsqueeze(-1)

        # print(f"rms:{rms.shape},raw:{raw.shape},x:{x.shape}")
        # y = self.scale[:x.shape[1],:]
        # print(f"\ny:{y.shape}\n")
        
        return self.scale[:x.shape[1],:].unsqueeze(0) * raw


class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # Linear transformation for query
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # Linear transformation for key
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # Linear transformation for value
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # Obtain rotary matrix for positional embeddings
        self.r = self.Rotary_embeddings(config['context_window'], config['d_model'])

    def Rotary_embeddings(self,context_window,embeddings_dim):

        r = torch.zeros((context_window,embeddings_dim,embeddings_dim),requires_grad=False)

        for pos in range(context_window):
            for i in range(embeddings_dim//2):
                phi = 10000**(-2*(i-1)/embeddings_dim)
                m_phi = pos*phi
                r[pos, 2 * i, 2 * i] = np.cos(m_phi)
                r[pos, 2 * i, 2 * i + 1] = -np.sin(m_phi)
                r[pos, 2 * i + 1, 2 * i] = np.sin(m_phi)
                r[pos, 2 * i + 1, 2 * i + 1] = np.cos(m_phi)

        return r
    
    def forward(self,
                x,
                return_attn_weights = False):
        
        b,m,d = x.shape
        # print(f"Rotary_embeddings:{b,m,d}")
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1),self.r[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1),self.r[:m])).transpose(0,1)

            
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones(m,m),diagonal = 0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        
        return activations
    
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        '''

        '''
        self.config = config 
        # Create a list of RoPEMaskedAttentionHead instances as attention heads
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])  # Linear layer after concatenating heads
        self.dropout = nn.Dropout(.1)  # Dropout layer

    def forward(self,x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads,dim = -1)
        x = self.linear(x)
        x = self.dropout(x)
        
        return x
    

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out