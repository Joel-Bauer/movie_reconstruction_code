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
model_list = [0, 1, 2, 3, 4, 5, 6] # 0,1,2,3,4,5,6 
number_models = np.array(model_list).shape[0]
animals = [0,1,2,3,4] # range(0,5) 
start_trial = [0]
end_trial = [10] # not incluing this one
random_trials = False # if true then random trials are chosen, default False
video_length = None # None. max is 300 but thats too much for my gpu
load_skip_frames = 0 # in case beginning of video should be skipped, default 0

# Reduced population size 
population_reduction = [0] # a fraction (0-1) of the population to drop, new sample for each trial (1/2,3/4,7/8)

# option to control parameters as inputs
parser = argparse.ArgumentParser(description='optional parameters as inputs. mouse,model_list')
parser.add_argument('--model_list', type=int, nargs='+', default=model_list, help='list of models to use for reconstruction 0 to 6')
parser.add_argument('--animals', type=int, nargs='+', default=animals, help='list of animals to reconstruct videos from [0,1,2,3,4]')
parser.add_argument('--start_trial', type=int, nargs='+', default=start_trial, help='first trail to reconstruction 0')
parser.add_argument('--end_trial', type=int, nargs='+', default=end_trial, help='last trial to reconstruct (not including this one) 10')
parser.add_argument('--population_reduction', type=float, default=population_reduction, help='fraction of population to reduce 0,0.5,0.75,0.875')
args = parser.parse_args()
model_list = np.array(args.model_list) 
number_models = np.array(model_list).shape[0]
animals = np.array(args.animals)
start_trial = args.start_trial[0]
end_trial = args.end_trial[0]
population_reduction = args.population_reduction[0]

# to optimize to the predicted responses rather than the gt responses
optimize_given_predicted_responese = False

# randomize neuron order or ground truth responese
randomize_neurons = False

# parameters for sliding window
subbatch_size = 32
minibatch = 8 # >2, <subbatch_size
epoch_number_first = 1000
n_steps = 1
epoch_reducer = 1 # eg. if 0.5 for every stride length take half the epochs of the previous stride
strides_all, epoch_switch = utils.stride_calculator(minibatch=minibatch,n_steps=n_steps,epoch_number_first=epoch_number_first,epoch_reducer=epoch_reducer)
print('strides_all: ' + str(strides_all) + '; epoch_switch: ' + str(epoch_switch))

# mask parameters
mask_update_th = 0.5 # [0:1], defaul 0.5
mask_eval_th = 1 # [0:1], default 1

# loss and regularizers
vid_init = 'gray' # 'noise', 'static_noise', 'gt_vid_first_1', 'gt_vid', 'gray'
lr = 1000 # usually between 100 and 10000
lr_warmup_epochs = 10 
loss_func = 'poisson' # 'poisson' or 'mse'
use_adam = True
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-8
with_gradnorm = True # default False
clip_grad = 1 # None, or threshold
eval_frame_skip = 32 # default 32

# parameters not used in the paper
response_dropout_rate = 0 # 0.5 # probability of dropout for responses and responses_predicted (matched units) durnig loss calcualtion (only for training)
drop_method='zero_pred_n_true' # 'zero_pred', 'zero_pred_n_true', 'set_pred_to_true
input_noise = 0 # 5
pix_decay_rate = 0 # try 0.005 with gray start

# save location
save_path_sufix = 'round1'

save_path = f'reconstructions/modelfold{model_list}_RFmapping_{save_path_sufix}/'

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

# create all predictors
# then all the predictors need to be loaded
predictor = [None]*len(model_path)
for n in range(0,len(predictor)):
    model_path_temp = model_path[n]
    print('predicting true resp with model: ', model_path_temp)
    predictor[n] = Predictor(model_path=model_path_temp, device=device, blend_weights="ones")# the input dims for this model are  (batch, channel, time, height, width): (32, 5, 16, 64, 64)...

        
# create stimulus set
# on or off pixels at each position with 1s prestim, 0.5s stimulus, 0.5s poststim
phases = 2
pixel_height = 36
pixel_width = 64
prestim = 15
stimulus_duration = 15
poststim = 15
video_length = prestim + stimulus_duration + poststim
stimulus_set = np.zeros((phases,pixel_height,pixel_width,pixel_height,pixel_width,video_length)) # dims are on/off, height_pix, width_pix, height, width,frames) 
print('stimulus_set shape: ' + str(stimulus_set.shape))

for stimulus_index in range(0,phases):
    for height_index in range(0,pixel_height):
        for width_index in range(0,pixel_width):
            background_value = 0.5
            if stimulus_index == 0:
                pix_value = 1
            elif stimulus_index == 1:
                pix_value = 0
            stimulus_set[stimulus_index,height_index,width_index,:,:,:] = background_value
            stimulus_set[stimulus_index,height_index,width_index,height_index,width_index,prestim:prestim+stimulus_duration] = pix_value
stimulus_set = stimulus_set*255

np.save(save_path+f'Sparse_noise_RFs_STIMULUS.npy',stimulus_set)   


responses_all = []
behavior_all = []
pupil_center_all = []
mouse_indexs = []


for mouse_index in range(0,len(animals)):
    # get mouse name            
    mouse = constants.index2mouse[mouse_index]
    print(f'\nmouse {mouse_index}: {mouse}')       
    
    behavior = np.nanmax(np.load(f'data/sensorium_all_2023/{mouse}/meta/statistics/behavior/all/mean.npy'),axis=1)
    behavior = behavior[:,None].repeat(video_length,axis=-1)
    pupil_center = np.nanmax(np.load(f'data/sensorium_all_2023/{mouse}/meta/statistics/pupil_center/all/mean.npy') ,axis=1)
    pupil_center = pupil_center[:,None].repeat(video_length,axis=-1)


    # prediction based on full video
    for stimulus_index in range(0,phases):
        responses_predicted_original = np.zeros((len(predictor), pixel_height, pixel_width, constants.num_neurons[mouse_index],3))
        
        for height_index in range(0,pixel_height):
            print(f'mouse {mouse_index} stimulus {stimulus_index} row {height_index}/36')
            for width_index in range(0,pixel_width):
                for model_idx in range(0,len(predictor)):
                    video_temp = np.pad(stimulus_set[stimulus_index,height_index,width_index],((14,14),(0,0),(0,0)), 'constant', constant_values=255/2)

                    responses = predictor[model_idx].predict_trial(
                        video=video_temp,
                        behavior=behavior,
                        pupil_center=pupil_center,
                        mouse_index=mouse_index)
                    
                responses_predicted_original[model_idx,height_index,width_index,:,0] = responses[:,0:prestim].mean(axis=-1)
                responses_predicted_original[model_idx,height_index,width_index,:,1] = responses[:,prestim:prestim+stimulus_duration].mean(axis=-1)
                responses_predicted_original[model_idx,height_index,width_index,:,2] = responses[:,prestim+stimulus_duration:].mean(axis=-1)

        output_dict = {'stimulus_set': [],
            'responses': responses_predicted_original, 
            'behavior': behavior, 
            'pupil_center': pupil_center, 
            'mouse_indexs': mouse_index}

        np.save(save_path+f'Sparse_noise_m{mouse_index}_phase{stimulus_index}_RFs.npy',output_dict)   



