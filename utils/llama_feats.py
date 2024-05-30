import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
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
