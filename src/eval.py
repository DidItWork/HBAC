import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

@torch.inference_mode()
def test(model=None,
          dataset=None,
          config=None):
    
    model.eval()
    
    batch_size = config.get("batch_size", 8)
    batch_size = config.get("batch_size", 32)
    shuffle = config.get("shuffle", True)

    test_dataloader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)

    model.train()

    