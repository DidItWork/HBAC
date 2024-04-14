import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .eval import test
from .loss import 

def train(model=None,
          train_dataset=None,
          test_dataset=None,
          optimizer = torch.optim.adam(),
          training_config=None,
          testing_config=None):
    
    """
    Training Function
    """

    assert(train_dataset!=None)
    
    assert(model!= None)

    #Training Params

    num_epochs = training_config.get("num_epochs", 5)
    to_gpu_keys = [""]
    batch_size = training_config.get("batch_size", 32)
    shuffle = training_config.get("shuffle", True)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size = batch_size,
                                  shuffle = shuffle)
    
    for epoch in range(num_epochs):

        for itr, batch in enumerate(train_dataloader):

            #Train One Iteration

            model.train()

            #Load data to GPU

            for key in to_gpu_keys:

                batch[key] = batch[key].cuda()

            predictions = model(batch)

            loss = loss_func(predictions)

            loss.backward()

            optimizer.step()
            
        
        #Test
        print("Testing")
        if test_dataset is not None:

            test(model,
                 test_dataset,
                 testing_config)
            
