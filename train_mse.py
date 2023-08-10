import dataloaders
import torch
import utils
import Model
import deeplearning
import numpy                as     np
from torchvision            import transforms
from PIL                    import Image
#%%
model = Model.ECCVGenerator()

dmodel, optimizer, report = deeplearning.train_MSE(
    test_ealuate=False,
    train_loader = dataloaders.train_dataloader,
    val_loader = dataloaders.test__dataloader,
    tets_loader = None,
    model=model,
    model_name="color_MSE_30_60",
    sleep=0.0,
    epochs=utils.num_epochs,
    learning_rate=utils.learning_rate,
    device='cuda',
    load_saved_model=False,
    ckpt_save_freq=10,
    ckpt_save_path  = './Model/',
    ckpt_path       = './Model/',
    report_path     = './Model/',
    total_iters=utils.total_iters,
    )

#%%
model = Model.ECCVGenerator()#.to('cuda')

model.load_state_dict(torch.load('Model/color_MSE_30_60.pt',map_location=torch.device('cpu')))
model.eval()


import pandas as pd
import numpy as np
import utils
_1=pd.read_csv('/content/Model/color_MSE_00_30_report.csv')
_2=pd.read_csv('/content/Model/color_MSE_30_60_report.csv')

df=pd.concat([_1, _2], ignore_index=True)

train=df.query('mode == "train"').query('batch_index == 40')
test=df.query('mode == "val"').query('batch_index == 10')

Model_name = 'MSE_'

utils.plot.result_plot(Model_name+"_loss","loss",
                        np.array(train['loss_batch']),
                        np.array(test['loss_batch']),
                        DPI=400)