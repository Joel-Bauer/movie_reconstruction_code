# standard
from pathlib import Path
import numpy as np
import torch 
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# lRomul
import argus
from src.data import get_mouse_data
from src.predictors import Predictor
from src import constants

# mine
import utils_reconstruction.image_similarity as im_sim
import utils_reconstruction.utils_reconstruction as utils
# from src.indexes import IndexesGenerator
# import tifffile
#import torchsummary
# import logging # use this instead of print


print('\n')
print('------------')

## parameters
track_itter = 10
plot_itter = track_itter*1
data_fold = 'fold_0' # choose test set, different to all chosen models
model_list = [0,1,2,3,4,5,6] # 0,1,2,3,4,5,6 # if this is more than one then the gradient is averaged across models at each step
number_models = np.array(model_list).shape[0]
animals = [0,1,2,3,4] # range(0,5) 
trial_ids = np.array([[0,1,2,10,4,5,6,7,8,11],
                    [10,1,2,3,11,5,6,7,8,9],
                    [10,1,14,3,4,5,6,15,16,9],
                    [0,1,2,3,10,5,6,7,8,9],
                    [0,1,2,3,4,5,6,7,8,9]]) # after replacing hashing error videos
trial_number = trial_ids.shape[1] # number of trials per animal, default 10
random_trials = False # if true then random trials are chosen, default False
video_length = 300 # None. max is 300 but thats too much for my gpu
load_skip_frames = 0 # in case beginning of video should be skipped, default 0

mouse_names = [
    "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20",
]

# option to control parameters as inputs
parser = argparse.ArgumentParser(description='optional parameters as inputs. mouse,model_list')
parser.add_argument('--model_list', type=int, nargs='+', default=model_list, help='list of models to use for reconstruction 0 to 6')
parser.add_argument('--animals', type=int, nargs='+', default=animals, help='list of animals to reconstruct videos from [0,1,2,3,4]')

args = parser.parse_args()
model_list = np.array(args.model_list) 
number_models = np.array(model_list).shape[0]
animals = np.array(args.animals)

# mask parameters
mask_eval_th = 0.5 # [0:1], only applying training mask

save_path = f'reconstructions/predict_from_recon/'
Path(save_path).mkdir(parents=True, exist_ok=True) 
print('save path:' + save_path)

#get available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ' + str(device))

# get a model
print('\nget a model')
model_path = [None]*7
model = [None]*number_models
# # old model, each model trained on all but its own fold, i.e. model0 trained on fold 1-6
# model_path[0] = Path('data/experiments/true_batch_001/fold_0/model-017-0.293169.pth') 
# model_path[1] = Path('data/experiments/true_batch_001/fold_1/model-017-0.295205.pth')
# model_path[2] = Path('data/experiments/true_batch_001/fold_2/model-017-0.294490.pth')
# model_path[3] = Path('data/experiments/true_batch_001/fold_3/model-017-0.292547.pth')
# model_path[4] = Path('data/experiments/true_batch_001/fold_4/model-017-0.291719.pth')
# model_path[5] = Path('data/experiments/true_batch_001/fold_5/model-017-0.291422.pth')
# model_path[6] = Path('data/experiments/true_batch_001/fold_6/model-017-0.292521.pth')

# new model, data fold is always fold 0
model_path[0] = Path('data/experiments/true_batch_002/fold_0/model-017-0.292118.pth')
model_path[1] = Path('data/experiments/true_batch_002/fold_1/model-017-0.291708.pth')
model_path[2] = Path('data/experiments/true_batch_002/fold_2/model-017-0.290428.pth')
model_path[3] = Path('data/experiments/true_batch_002/fold_3/model-017-0.291853.pth')
model_path[4] = Path('data/experiments/true_batch_002/fold_4/model-017-0.291103.pth')
model_path[5] = Path('data/experiments/true_batch_002/fold_5/model-017-0.291734.pth')
model_path[6] = Path('data/experiments/true_batch_002/fold_6/model-017-0.291974.pth')

# load all selected models (selection is defined by model_list)
for n in range(0,len(model_list)):
    print('loading model for reconstruction: ', model_path[model_list[n]])
    model[n] = argus.load_model(model_path[model_list[n]], device=device, optimizer=None, loss=None)
    model[n].eval() # the input dims for this model are  (batch, channel, time, height, width): (32, 5, 16, 64, 64)...

# then only the models used for reconstuction are loaded
predictor = [None]*number_models
for n in range(0,len(predictor)):
    model_path_temp = model_path[model_list[n]]
    print('predicting true resp with model: ', model_path_temp)
    predictor[n] = Predictor(model_path=model_path_temp, device=device, blend_weights="ones")        

num_neurons = [7863, 7908, 8202, 7939, 8122]
video_gt=np.nan*np.ones((len(animals),len(range(trial_number)),300,36,64))
video_pred=np.nan*np.ones((model_list.size,len(animals),len(range(trial_number)),300,36,64))
behavior=np.nan*np.ones((len(animals),len(range(trial_number)),2,300))
pupil_center=np.nan*np.ones((len(animals),len(range(trial_number)),2,300))
responses=torch.from_numpy(np.nan*np.ones((len(animals),len(range(trial_number)),np.max(num_neurons),300)))

mask=np.nan*np.ones((len(animals),64,64))

eval_frame_skip = 30 # default 30

for model_n in model_list:
    for mouse in animals:
        for trial in range(trial_number):
            datapath=f'reconstructions/modelfold[{model_n}]_datafold_0_pop100_hpc_round5/reconstruction_summary_m{mouse}_t{trial_ids[mouse,trial]}.npy'
            print(datapath)
            data = np.load(datapath, allow_pickle=True).item()
            if model_n==model_list[1]:
                video_gt[mouse,trial] = data['video_gt']
                behavior[mouse,trial] = data['behavior']
                pupil_center[mouse,trial] = data['pupil_center']
                responses[mouse,trial,0:num_neurons[mouse],:] = data['responses']
            
            video_pred[model_n,mouse,trial] = data['video_pred']
            mask[mouse] = data['mask'][:,:]

video_pred = np.nanmean(video_pred,axis=0)  # average across models       
print('video_pred shape: ', video_pred.shape)   

def get_losses(video,responese_original,eval_frame_skip=30):
    responses_predicted = np.zeros((num_neurons[mouse_index], video_length, len(predictor)))
    for n in range(0,len(predictor)):
        responses_predicted[:,:,n] = predictor[n].predict_trial(
            video=video,
            behavior=behavior[mouse_index,trial_n],
            pupil_center=pupil_center[mouse_index,trial_n],
            mouse_index=mouse_index)
        
    responses_predicted = torch.from_numpy(responses_predicted.mean(axis=2).copy()).to(device)
    poisson_loss_gt = utils.response_loss_function(responses_predicted[:,eval_frame_skip:].to(device),
                                    responese_original[:,eval_frame_skip:].clone().detach().to(device), loss_func='poisson')
    mse_loss_gt = utils.response_loss_function(responses_predicted[:,eval_frame_skip:].to(device),
                                    responese_original[:,eval_frame_skip:].clone().detach().to(device), loss_func='mse')
    response_corr_gt = np.corrcoef(responses_predicted[:,eval_frame_skip:].cpu().detach().numpy().flatten(), 
                                responese_original[:,eval_frame_skip:].cpu().detach().numpy().flatten())[0,1]
    video_first_frame = video[:,:,eval_frame_skip] # first frame of video for plotting
    return responses_predicted.to('cpu'), poisson_loss_gt, mse_loss_gt, response_corr_gt, video_first_frame

## loop through trials
responses_pred = np.zeros_like(responses)
responses_pred_masked = np.zeros_like(responses)
responses_pred_recon = np.zeros_like(responses)
responses_pred_recon_contrast_lum = np.zeros_like(responses)

poisson_loss_gt = np.zeros((len(animals),trial_number))
mse_loss_gt = poisson_loss_gt.copy()
response_corr_gt = poisson_loss_gt.copy()
first_frame_gt = np.zeros((len(animals),trial_number,36,64))

poisson_loss_masked = poisson_loss_gt.copy()
mse_loss_masked = poisson_loss_gt.copy()
response_corr_masked = poisson_loss_gt.copy()
first_frame_gt_masked = np.zeros((len(animals),trial_number,36,64))

poisson_loss_recon = poisson_loss_gt.copy()
mse_loss_recon = poisson_loss_gt.copy()
response_corr_recon = poisson_loss_gt.copy()
first_frame_recon = np.zeros((len(animals),trial_number,36,64))

poisson_loss_recon_contrast_lum = poisson_loss_gt.copy()
mse_loss_recon_contrast_lum = poisson_loss_gt.copy()
response_corr_recon_contrast_lum = poisson_loss_gt.copy()
first_frame_recon_contrast_lum = np.zeros((len(animals),trial_number,36,64))

for mouse_index in animals:
    
    # load mask
    mask_eval = np.where(mask[mouse_index] >= mask_eval_th,1,0)
    mask_eval_expanded = mask_eval[14:14+36,:,None].repeat(300,axis=2)
    print('mask shape: ', mask_eval_expanded.shape)
    
    for trial_n in range(trial_number):
        population_mask = None        
        
        # get current responses
        responses_temp = responses[mouse_index,trial_n,:num_neurons[mouse_index],:].to(device)
        
        # prediction based on full video
        video_original = np.moveaxis(video_gt[mouse_index,trial_n],0,-1)
        responses_pred[mouse_index,trial_n,:num_neurons[mouse_index]], poisson_loss_gt[mouse_index,trial_n],mse_loss_gt[mouse_index,trial_n],response_corr_gt[mouse_index,trial_n],first_frame_gt[mouse_index,trial_n]=get_losses(video_original,responses_temp,eval_frame_skip=eval_frame_skip)
        print( f'gt test losses (poisson, mse, corr): {poisson_loss_gt[mouse_index,trial_n]}, {mse_loss_gt[mouse_index,trial_n]}, {response_corr_gt[mouse_index,trial_n]}')
        
        # prediction based on video with mask
        video_original_masked= np.where(mask_eval_expanded,np.moveaxis(video_gt[mouse_index,trial_n],0,-1),255//2)
        responses_pred_masked[mouse_index,trial_n,:num_neurons[mouse_index]], poisson_loss_masked[mouse_index,trial_n],mse_loss_masked[mouse_index,trial_n],response_corr_masked[mouse_index,trial_n],first_frame_gt_masked[mouse_index,trial_n]=get_losses(video_original_masked,responses_temp,eval_frame_skip=eval_frame_skip)
        print( f'masked test losses (poisson, mse, corr): {poisson_loss_masked[mouse_index,trial_n]}, {mse_loss_masked[mouse_index,trial_n]}, {response_corr_masked[mouse_index,trial_n]}')
        
        # prediction based on reconstructed video
        video_recon = np.where(mask_eval_expanded,np.moveaxis(video_pred[mouse_index,trial_n],0,-1),255//2)
        responses_pred_recon[mouse_index,trial_n,:num_neurons[mouse_index]], poisson_loss_recon[mouse_index,trial_n],mse_loss_recon[mouse_index,trial_n],response_corr_recon[mouse_index,trial_n],first_frame_recon[mouse_index,trial_n]=get_losses(video_recon,responses_temp,eval_frame_skip=eval_frame_skip)
        print( f'recon test losses (poisson, mse, corr): {poisson_loss_recon[mouse_index,trial_n]}, {mse_loss_recon[mouse_index,trial_n]}, {response_corr_recon[mouse_index,trial_n]}')
        
        # prediction based on contrast and luminance matched video
        video_recon_contrast_lum,_ = utils.luminance_contrast_match(video_gt[mouse_index,trial_n],video_pred[mouse_index,trial_n],mask_eval_expanded[:,:,0],mask_eval_th)
        video_recon_contrast_lum = np.moveaxis(video_recon_contrast_lum,0,-1)
        responses_pred_recon_contrast_lum[mouse_index,trial_n,:num_neurons[mouse_index]], poisson_loss_recon_contrast_lum[mouse_index,trial_n],mse_loss_recon_contrast_lum[mouse_index,trial_n],response_corr_recon_contrast_lum[mouse_index,trial_n],first_frame_recon_contrast_lum[mouse_index,trial_n]=get_losses(video_recon_contrast_lum,responses_temp,eval_frame_skip=eval_frame_skip)
        print( f'recon contrast lum test losses (poisson, mse, corr): {poisson_loss_recon_contrast_lum[mouse_index,trial_n]}, {mse_loss_recon_contrast_lum[mouse_index,trial_n]}, {response_corr_recon_contrast_lum[mouse_index,trial_n]}')

recon_dict = {'responses': responses,
              
                    'responses_pred': responses_pred,
                    'poisson_loss_gt': poisson_loss_gt,
                    'mse_loss_gt': mse_loss_gt,
                    'response_corr_gt': response_corr_gt,
                    
                    'responses_masked': responses_pred_masked,
                    'poisson_loss_masked': poisson_loss_masked,
                    'mse_loss_masked': mse_loss_masked,
                    'response_corr_masked': response_corr_masked,
                    
                    'responses_pred_recon': responses_pred_recon,
                    'poisson_loss_recon': poisson_loss_recon,
                    'mse_loss_recon': mse_loss_recon,
                    'response_corr_recon': response_corr_recon,
                    
                    'responses_pred_recon_contrast_lum': responses_pred_recon_contrast_lum,
                    'poisson_loss_recon_contrast_lum': poisson_loss_recon_contrast_lum,
                    'mse_loss_recon_contrast_lum': mse_loss_recon_contrast_lum,
                    'response_corr_recon_contrast_lum': response_corr_recon_contrast_lum}
                    
np.save(f'reconstructions/activity_pred_from_recon_versions.npy',recon_dict)   

        