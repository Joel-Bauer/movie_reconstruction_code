from pathlib import Path
import numpy as np
import cv2
import torch 
import argus
from src.data import get_mouse_data
from src.predictors import Predictor
from src.inputs import get_inputs_processor
from src.responses import get_responses_processor
import matplotlib.pyplot as plt
from src import constants
from tqdm import tqdm
import utils_reconstruction.utils_reconstruction as utils

print('\n')
print('------------')

# def loss function
def loss_function_with_mask(responses_predicted,responses,mask=None):
    loss = torch.nn.MSELoss()
    if mask is not None:
        responses_predicted = responses_predicted*(1-mask.mean())
    # loss = torch.nn.PoissonNLLLoss(log_input=False, full=False)
    return loss(responses_predicted.float(),responses.float())

# get a model
print('get a model')
model_path = Path('data/experiments/true_batch_001/fold_1/model-017-0.295205.pth')
model = argus.load_model(model_path, device="cuda:0", optimizer=None, loss=None)
model.eval() # the input dims for this model are  (batch, channel, time, height, width): (32, 5, 16, 64, 64)...
print(model_path)

# choose datafold
datafold='fold_1' # same as model
skip_frames = 50
length = 32 # max is 300 but thats too much for my gpu

# get data cycler
# get a batch
print('get a batch')
for mouse_index in range(0,5):
    mouse_index = mouse_index
    mouse = constants.index2mouse[mouse_index]
    mouse_data = get_mouse_data(mouse=mouse, splits=[datafold])
    trial_data_all=mouse_data['trials']
    
    # define preditor which tracks gradients
    predictor_withgrads = utils.Predictor_JB(model, mouse_index, withgrads=True, dummy=False)    
    
    # initialize mask
    mask = torch.rand((1,length,64,64)).to(model.device)*0.05
    
    lr = 10
    epochs = 1000
    progress_bar = tqdm(range(epochs))
    for i in progress_bar:  
    # for rand_trial in range(5):
        trial = np.random.randint(0,len(trial_data_all)-1, 1)[0]
        trial2 = np.random.randint(0,len(trial_data_all)-1, 1)[0]
            
        inputs, responses,_,_,_ = utils.load_trial_data(trial_data_all, model, trial, skip_frames, length=length, print_dims=False)
        inputs=inputs.to(model.device)
        responses=responses.to(model.device)
        
        inputs2, _,_,_,_ = utils.load_trial_data(trial_data_all, model, trial2, skip_frames, length=length, print_dims=False)
        alternative_video=inputs2[0:1,:,:,:].to(model.device)
        
        inputs_withmask = utils.cat_video_behaviour_mask(inputs[None,0:1,:,:,:],mask[None,0:1,:,:,:],inputs[None,1:,:,:,:]).detach().requires_grad_(True)
        # run model
        responses_predicted = predictor_withgrads(inputs[None,:,:,:,:],with_mask=False, background_video=None)
        responses_predicted_masked = predictor_withgrads(inputs_withmask,with_mask=True, background_video=alternative_video)
        
        # get gradients
        # loss = loss_function_with_mask(responses_predicted_masked,responses_predicted)
        loss = loss_function_with_mask(responses_predicted_masked,responses.clone().detach().to(model.device))
        loss.backward()
        gradients = inputs_withmask.grad    
        gradients = gradients / (torch.norm(gradients) + 1e-6) # normalize gradients
        
        # clip gradients
        gradients = torch.clip(gradients, -1, 1)  

        # mean gradients
        gradients = gradients.mean(axis=2, keepdim=True)
        
        # smooth video gradients
        sigma_blur_time = 0 # so far best results with 1 but it might be causing artefacts
        sigma_blur_spatial = 5
        gradients = utils.apply_gaussian_blur_3d(gradients[0], sigma_blur_time, sigma_blur_spatial,factor=0).unsqueeze(0)
        gradients=gradients.repeat(1,1,inputs.shape[1],1,1)
        
        # update video and mask 
        mask = torch.add(mask, -lr*gradients[0,1:2])

        # save first loss
        if i == 0:
            loss_init = loss.item()
        
        # Update the progress bar and print loss
        progress_bar.set_postfix(variable_message=f'loss: {loss.item():.3f} / {loss_init:.0f}', refresh=True)
        progress_bar.update()

        if i == 0 or i % 50 == 0 or i == epochs-1:
            # 2d mask
            mask2d = mask[0,0].cpu().detach().numpy()
            masksum=mask2d.sum().sum()
            
            # save mask
            fig, axs = plt.subplots(1,1, figsize=(5, 5))
            axs.imshow(mask[0,0].cpu().detach().numpy(),vmin=0,vmax=1)
            fig.colorbar(axs.imshow(mask[0,0].cpu().detach().numpy(),vmin=0,vmax=1), ax=axs)    
            axs.axis('off')
            axs.set_title(f'mask for m{mouse_index}, sum: {masksum}')
            fig.savefig(f'reconstructions/masks/mask_summary_m{mouse_index}.png')
            plt.close(fig)
            
            # save mask as variable
            np.save(f'reconstructions/masks/mask_m{mouse_index}.npy',mask[0,0].cpu().detach().numpy())
                    