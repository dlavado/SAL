

from collections import OrderedDict
from typing import List, Tuple
import torch
from torch.autograd import Variable
import numpy as np
import pytorch_lightning as pl

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
from core.models.lit_modules.lit_wrapper import LitWrapperModel




# MLP shell for PINN architecture

class MLP(torch.nn.Module):
    
    def __init__(self, layers: List[int], batch_norm: bool = False, dropout: float = 0.0):
        super(MLP, self).__init__()
        
        # Parameters
        self.depth = len(layers) - 1
        
        
        # Construct MultiLayer Perceptron (MLP)
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if batch_norm:
                layer_list.append(('batch_norm_%d' % i, torch.nn.BatchNorm1d(layers[i+1])))
            
            layer_list.append(('activation_%d' % i, torch.nn.ReLU()))
            
            layer_list.append((f"Dropout {i}", torch.nn.Dropout1d(p=0.1)))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.model = torch.nn.Sequential(layerDict).requires_grad_(True)

    def forward(self, x):
        # print(f"--Loc: MLP.forward()--")
        # print(f"x shape: {x.shape}; req_grad: {x.requires_grad}")
        x = torch.flatten(x, start_dim=1)
        # print(f"x flattened shape: {x.shape}; req_grad: {x.requires_grad}")
        out = self.model(x)
        # print(f"out shape: {out.shape}; req_grad: {out.requires_grad}")
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")

        return out
    
    def prediction(self, model_output):
        return model_output
    

class Lit_PINN(LitWrapperModel):

    def __init__(self, layers, batch_norm, dropout, optimizer_name: str, learning_rate=0.01, metric_initializer=None, **kwargs):
        model = MLP(layers, batch_norm, dropout)
        criterion = torch.nn.MSELoss()
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)

    def preprocess_batch(batch:Tuple[torch.Tensor, torch.Tensor]):
        # batch of orbit dataset is a tuple with (time_window of shape [T, feat_size + 1], the T+1 time step of size [feat_size + 1]); the `1` is for the time feature
        x, y = batch
        x, time = x[:, :, :-1], x[:, :, -1, None]
        time = time.requires_grad_()
        x = x.requires_grad_()
        y = y[:, :-1]
        return x, time, y


    def forward(self, x, t):
        # print(f"--Loc: Lit_PINN.forward()--")
        # print(f"x shape: {x.shape}; req_grad: {x.requires_grad}")
        # print(f"t shape: {t.shape}; req_grad: {t.requires_grad}")
        # print(f"cat: {torch.cat((x, t), dim=-1).shape}; req_grad: {torch.cat((x, t), dim=-1).requires_grad}")
        # input shape: (batch_size, window_size, feat_size + 1), where the +1 is for time
        return self.model(torch.cat([x, t], dim=-1))




if __name__ == '__main__':

    from core.data_modules.orbit_dataset import OrbitDataModule
    from my_utils import constants as const
    from core.models.lit_modules.lit_wrapper import Lit_FixedPenalty_PINN_Wrapper, LitWrapperModel


    orbit_dm = OrbitDataModule(
        const.ORBIT_DATASET_PATH,
        window_size=5,
        batch_size=4
    )

    # 3-body problem
    # tensors with shape: x (B, window_size, feat_size), t (B, window_size, 1), y (B, window_size, 1)
    # x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32, requires_grad=True)
    # t = torch.tensor([[[0], [1], [2]]], dtype=torch.float32, requires_grad=True)
    # y = torch.tensor([[[1], [2], [3]]], dtype=torch.float32, requires_grad=True)

    # model = Lit_PINN([torch.numel(x) + torch.numel(t), 10, 10, 10, 10, 10, 10, 10, 10, 1], 'Adam', learning_rate=0.01)

    model = Lit_PINN(
        [7*5, 5, 5, 5, 6], 'adam'
    )

    orbit_dm.setup("fit")
    dl = orbit_dm.train_dataloader()

    batch = dl.__iter__().__next__()
    x, y = batch
    x, t = x[:, :, :-1], x[:, :, -1, None]
    x = x.requires_grad_()
    t = t.requires_grad_()

    out = model(x, t)
    print(out)

        
