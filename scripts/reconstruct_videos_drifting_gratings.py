from pathlib import Path
import numpy as np
import torch 
import argus
from src.data import get_mouse_data
from src.predictors import Predictor
import utils_reconstruction.image_similarity as im_sim
import time
# from src.indexes import IndexesGenerator
import matplotlib.pyplot as plt
# import tifffile
#import torchsummary
from src import constants
from tqdm import tqdm
import utils_reconstruction.utils_reconstruction as utils
# import logging # use this instead of print



print('\n')
print('------------')

## parameters
track_itter = 10
plot_itter = track_itter*1
data_fold = 'fold_0' # choose test set, different to all chosen models
model_response_pred_list = np.array([0,1,2,3,4,5,6]) # 1,2,3,4,5,6 # used for estimating gt true responses
model_list = [0] # 0,1,2,3,4,5,6 
number_models = model_list.shape[0]
animals = [0] # range(0,1) 
reprange = range(0,4) # range(0,4)
video_length = None # None max is 300 but thats too much for my gpu

# option to control parameters as inputs
import argparse
parser = argparse.ArgumentParser(description='optional parameters as inputs. mouse,model_list')
parser.add_argument('--model_list', type=int, nargs='+', default=model_list, help='list of models to use for reconstruction [0-6]')
parser.add_argument('--animals', type=int, nargs='+', default=animals, help='list of models to use for reconstruction [0-6]')
args = parser.parse_args()
model_list = np.array(args.model_list) 
animals = np.array(args.animals)

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
response_dropout_rate = 0 # 0.5 # probability of dropout for responses and responses_predicted (matched units) durnig loss calcualtion (only for training)
drop_method='zero_pred_n_true' # 'zero_pred', 'zero_pred_n_true', 'set_pred_to_true
input_noise = 0 # 5
pix_decay_rate = 0 # try 0.005 with gray start

#get available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ' + str(device))

# get a model
print('\nget a model')
model_path = [None]*7
# # model_path[0] = Path('data/experiments/true_batch_001/fold_0/model-017-0.293169.pth') # data held out as test set
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

# model_cpu = [None]*len(model_response_pred_list)
# for n in range(0,len(model_response_pred_list)):
#     print(model_path[model_response_pred_list[n]])
#     model_cpu[n] = argus.load_model(model_path[model_response_pred_list[n]], device='cpu', optimizer=None, loss=None)
#     model_cpu[n].eval() # the input dims for this model are  (batch, channel, time, height, width): (32, 5, 16, 64, 64)...

# get drifting gratings noise stimulus
data = np.load('utils_reconstruction/grating_movies.npz', allow_pickle=True)
vids = data['video']
vids = np.moveaxis(vids, [1,2], [2,1]) # switch spatial and temporal dimensions
print(vids.shape)
print('Drifting grating stimulus shape: ', vids.shape) #  reps: 5, spatial length constant: 7, temporal length constant: 7, frames: 60, h: 36,w: 64
video_length = vids.shape[-3]

# loop order (last to first priority)
# reps, models, mouse, spatial, length constant, temporal length constant
for rep in reprange: # predetermined to slip across nodes but full range would be range(0,vids.shape[0]):
    for current_model in range(0,number_models):
        # load model
        print('model: ', model_path[model_list[current_model]])
        model = argus.load_model(model_path[model_list[current_model]], device=device, optimizer=None, loss=None)
        
        # save location
        save_path = f'reconstructions/modelfold{model_list[current_model]}_round4_Drifting_gratings/'
        Path(save_path).mkdir(parents=True, exist_ok=True) 
        print('save path:' + save_path)

        for mouse_index in animals:
            for temporal_length_constant in range(vids.shape[2]-1,-1,-1): #first slow then fast
                for spatial_length_constant in range(0,vids.shape[1]): # first small then big
                    print(f'model fold {model_list[current_model]} mouse {mouse_index} dir {rep} sf {spatial_length_constant} tf {temporal_length_constant}')
                    
                    #load mask
                    mask = np.load(f'reconstructions/masks/mask_m{mouse_index}.npy')
                    mask_update = torch.tensor(np.where(mask >= mask_update_th,1,0)).to(device)
                    mask_eval = torch.tensor(np.where(mask >= mask_eval_th,1,0)).to(device)
                    print('mask shape: ', mask.shape)      
                    
                    # prepare mask
                    mask_update_expanded = mask_update.repeat(1,1,video_length,1,1)
                    mask_eval_expanded = mask_eval.repeat(1,1,video_length,1,1)
                            
                    # get mouse name            
                    mouse = constants.index2mouse[mouse_index]
                    print(f'\nmouse {mouse_index}: {mouse}')                    
                    
                    # get video, behaviour and pupil position
                    video = np.transpose(vids[rep,spatial_length_constant,temporal_length_constant,:,:,:],(1,2,0)) # time last
                    behavior = np.nanmax(np.load(f'data/sensorium_all_2023/{mouse}/meta/statistics/behavior/all/mean.npy'),axis=1)
                    behavior = behavior[:,None].repeat(video.shape[-1],axis=-1)
                    pupil_center = np.nanmax(np.load(f'data/sensorium_all_2023/{mouse}/meta/statistics/pupil_center/all/mean.npy') ,axis=1)
                    pupil_center = pupil_center[:,None].repeat(video.shape[-1],axis=-1)
                    
                    print('video shape: ', video.shape)
                    print('behavior shape: ', behavior.shape)
                    print('pupil_center shape: ', pupil_center.shape)
                    
                    video_forcat = np.pad(video[None,:,:,:],((0,0),(14,14),(0,0),(0,0)), 'constant', constant_values=125)
                    behavior_forcat = behavior[:,None,None,:].repeat(video_forcat.shape[-2],axis=-2).repeat(video_forcat.shape[-3],axis=-3)
                    pupil_center_forcat = pupil_center[:,None,None,:].repeat(video_forcat.shape[-2],axis=-2).repeat(video_forcat.shape[-3],axis=-3)
                    inputs = np.concatenate((video_forcat,behavior_forcat,pupil_center_forcat),axis=0).swapaxes(-1,-3)
                    inputs = torch.tensor(inputs).to(device).float()
                    video_forcat = []; behavior_forcat = [];pupil_center_forcat = []
                    print('inputs shape: ', inputs.shape)
                    
                    # move to torch
                    # video = torch.tensor(video).to(device).float()
                    inputs = torch.tensor(inputs).float()
                    
                    # Get predicted responses
                    predictor_all_models = [None]*len(model_response_pred_list)
                    responses_predicted_original = np.zeros((len(model_response_pred_list),constants.num_neurons[mouse_index], video_length))
                    for n in range(0,len(model_response_pred_list)):
                        predictor_all_models[n] = Predictor(model_path=model_path[model_list[current_model]], device=device, blend_weights="ones")
                        responses_predicted_original[n] = predictor_all_models[n].predict_trial(
                            video=video,
                            behavior=behavior,
                            pupil_center=pupil_center,
                            mouse_index=mouse_index)
                    responses_predicted_original = torch.tensor(responses_predicted_original.mean(axis=0)).float()
                    predictor_all_models = [] # clear models from gpu
                    # since we are only working with simulated data, this is now our ground truth
                    
                    # get current model predictor (no grad), needed later for loss tracking
                    predictor = Predictor(model_path=model_path[model_list[current_model]], device=device, blend_weights="ones")

                    # define preditor which tracks gradients
                    predictor_withgrads = utils.Predictor_JB(model, mouse_index, withgrads=True, dummy=False)
                    print('\nskipping test of gradient tracking')

                    # plot the responses and the predicted responses to check
                    print('responses range: ' + str(responses_predicted_original.min().cpu().detach().numpy()) + ' to ' + str(responses_predicted_original.max().cpu().detach().numpy()))
                    print('vid range: ' + str(inputs[0,:].min().cpu().detach().numpy()) + ' to ' + str(inputs[0,:].max().cpu().detach().numpy()))

                    ### export figure containing first frame of video, behaviour, responeses and predictions
                    fig, axs = plt.subplots(4,4, figsize=(20, 20))
                    fig.suptitle(f'mouse {mouse_index} r{rep} s{spatial_length_constant} t{temporal_length_constant}', fontsize=16)
                    axs[0,0].imshow(np.concatenate((video[:,:,0],video[:,:,-1]),axis=1),cmap='gray')
                    axs[0,0].axis('off')
                    axs[0,0].set_title('video frame 0 and ' + str(video.shape[2]))
                    axs[1,0].plot(behavior[:,:].T)
                    axs[1,0].set_title('behaviour')
                    axs[2,0].plot(pupil_center[:,:].T)
                    axs[2,0].set_title('pupil position')
                    axs[2,1].imshow(responses_predicted_original.cpu().detach().numpy(),aspect='auto',vmin=0,vmax=10)
                    axs[2,1].set_title('predicted responses')
                    axs[0,2].plot(np.sum(np.abs(np.diff(video, axis=2)),axis=(0,1)))
                    axs[0,2].set_title('video motion energy')
                    fig.savefig(save_path+f'reconstruction_summary_m{mouse_index}_d{rep}s{spatial_length_constant}t{temporal_length_constant}.png')
        
        
                    ### lets get started ###
                    # initialize video
                    print('\n')
                    video_pred = utils.init_weights(inputs,vid_init,device)

                    ## loop to optimize video
                    print('\n')
                    
                    # prepare im_sim vars
                    video_corr = []
                    video_RMSE = []
                    video_itter = []
                    loss_all = []
                    response_corr = []
                        
                    progress_bar = tqdm(range(epoch_switch[-1]))
                    start_training_time = time.time()
                    for i in progress_bar:
                        # get subbatch_shift = strides_all[current epoch switch]
                        for n in range(0,n_steps):
                            if i < epoch_switch[n]:
                                subbatch_shift = strides_all[n]
                                break
                                        
                        number_of_subbatches = 2+(inputs.shape[1]-subbatch_size)//subbatch_shift
                        
                        # initialize gradient accumulator
                        gradients_fullvid = torch.zeros_like(video_pred).repeat(number_of_subbatches,1,1,1,1).to(device).requires_grad_(False) # subbatch, channel, time, height, width
                        gradients_fullvid = gradients_fullvid.fill_(float('nan'))
                        gradnorm = torch.zeros(number_of_subbatches).to(device)
                        
                        
                        for subbatch in range(0,number_of_subbatches,1):
                            # torch.cuda.empty_cache() 
                            if subbatch == number_of_subbatches-1:
                                start_frame = inputs.shape[1]-subbatch_size
                                end_frame = inputs.shape[1]
                            else:
                                start_frame = subbatch*subbatch_shift
                                end_frame = subbatch*subbatch_shift+subbatch_size
                            subbatch_frames = range(start_frame,end_frame,1)

                            # add input noise
                            if input_noise>0:
                                added_video_noise = torch.randn_like(video_pred[:,:,subbatch_frames,:,:])*input_noise
                            else:
                                added_video_noise = torch.zeros_like(video_pred[:,:,subbatch_frames,:,:])
                            
                            # add video to behaviour
                            input_prediction = utils.cat_video_behaviour(video_pred[:,:,subbatch_frames,:,:]+added_video_noise,inputs[None,1:,subbatch_frames,:,:]).detach().requires_grad_(True)    
                            
                            responses_predicted_new = predictor_withgrads(input_prediction)
                            
                            # get a gradients
                            loss = utils.response_loss_function(responses_predicted_new,
                                                            responses_predicted_original[:,subbatch_frames].clone().detach().to(device).requires_grad_(True), mask=None)
                            loss.backward()
                            gradients = input_prediction.grad   
                            
                            # normalize gradients
                            gradnorm[subbatch] = torch.norm(gradients)
                            if with_gradnorm:
                                gradients = gradients / (gradnorm[subbatch] + 1e-6)
                            else:
                                gradients = gradients*100
                            
                            # add to gradient of video channel to accumulator
                            gradients_fullvid[subbatch,:,subbatch_frames,:,:] = gradients[:,0:1,:,:,:]
                                                                                    
                        # average gradients across subbatches and model pedictions excluding 0s
                        gradients_fullvid = torch.nanmean(gradients_fullvid, axis=0, keepdim=True) # mean across all subbatches
                        
                        # mask gradients
                        gradients_fullvid = gradients_fullvid*mask_update_expanded
                        
                        # clip gradients
                        if clip_grad is not None:
                            gradients_fullvid = torch.clip(gradients_fullvid, -1*clip_grad, clip_grad)
                        
                        # GD or Adam based weight update
                        if i > 0: # to track where we start from we just skip the first update
                            if lr_warmup_epochs>0 and i<lr_warmup_epochs:
                                lr_current = lr*min(1,i/lr_warmup_epochs)
                            else:
                                lr_current = lr
                            
                            if use_adam == False:
                                video_pred = torch.add(video_pred, -lr_current*gradients_fullvid[0:1,0:1])
                            else:
                                # adam optimizer
                                if i == 1:
                                    m = torch.zeros_like(gradients_fullvid)
                                    # v = torch.zeros_like(gradients_fullvid)
                                lr_t = lr_current * (1-adam_beta2**(i+1))**0.5 / (1-adam_beta1**(i+1))
                                m = adam_beta1*m + (1-adam_beta1)*gradients_fullvid
                                m_hat = m/(1-adam_beta1**(i+1))
                                # v = adam_beta2*v + (1-adam_beta2)*gradients_fullvid**2
                                # v_hat = v/(1-adam_beta2**(i+1))
                                video_pred = torch.add(video_pred, -lr_t*m_hat) # no second order momentum          
                                # video_pred = torch.add(video_pred, -lr_t*m_hat/(v_hat**0.5+adam_eps)) # second order momentum

                        # initial max min gradients
                        if i == 0:
                            maxgrad = np.max([np.abs(gradients_fullvid[0,0].mean(axis=(0)).cpu().detach().numpy().min()), np.abs(gradients_fullvid[0,0].mean(axis=(0)).cpu().detach().numpy().max())])

                        # clip video
                        video_pred = torch.clip(video_pred, 0, 255)
                        
                        # pixel decay (like weight decay)
                        if pix_decay_rate>0:
                            video_pred = ((video_pred-255/2)*(1-pix_decay_rate))+(255/2)
                        
                        # detach and requires grad, to reset
                        video_pred=video_pred.detach().requires_grad_(True)
                        
                        # get first iter loss
                        if i == 0:
                            loss_init = loss.item()

                        # every track_itter epochs save the video
                        if i==0 or i % track_itter == 0 or i == epoch_switch[-1]-1:
                            progress_bar.set_postfix(variable_message=f'loss: {loss.item():.3f} / {loss_init:.0f}', refresh=True)
                            progress_bar.update()
                            ground_truth = np.moveaxis(video,[2],[0])
                            reconstruction = video_pred[0,0].cpu().detach().numpy()
                            reconstruction = reconstruction[:,14:14+36,:]
                            mask_cropped = mask_eval[14:14+36,:].cpu().detach().numpy()
                            reconstruction_masked = reconstruction*mask_cropped + np.ones_like(reconstruction)*(1-mask_cropped)*255/2
                            
                            # full loss
                            responses_predicted_full = predictor.predict_trial(
                                video=np.moveaxis(reconstruction_masked,[0],[2]),
                                behavior=behavior,
                                pupil_center=pupil_center,
                                mouse_index=mouse_index)
                            
                                
                            loss_full = utils.response_loss_function(torch.from_numpy(responses_predicted_full[:,eval_frame_skip:]).to(device),
                                                                responses_predicted_original[:,eval_frame_skip:].clone().detach().to(device),
                                                                mask=None)   
                            loss_all.append(loss_full.item())
                            
                            # save movie as tiff
                            concat_video = utils.save_tif(ground_truth, reconstruction, save_path+f'optimized_input_m{mouse_index}_r{rep}s{spatial_length_constant}t{temporal_length_constant}.tif',mask=mask_cropped)
                                                            
                            # calculate similarity metrics
                            video_corr.append(im_sim.reconstruction_video_corr(ground_truth[eval_frame_skip:], reconstruction[eval_frame_skip:], mask_cropped))
                            video_RMSE.append(im_sim.reconstruction_video_RMSE(ground_truth[eval_frame_skip:], reconstruction[eval_frame_skip:], mask_cropped))
                            video_itter.append(i)    
                            response_corr.append(np.corrcoef(responses_predicted_original[:,:].cpu().detach().numpy().flatten(), 
                                                            responses_predicted_full.flatten())[0,1])
                            motion_energy_gt = im_sim.video_energy(ground_truth[eval_frame_skip:], mask_cropped)
                            motion_energy_recon = im_sim.video_energy(reconstruction[eval_frame_skip:], mask_cropped)
                            
                            
                        if i==0 or i % plot_itter == 0 or i == epoch_switch[-1]-1:         
                            axs[0,1].clear()
                            axs[0,1].imshow(np.concatenate((concat_video[0],concat_video[-1]),axis=1),cmap='gray',vmin=0,vmax=255)
                            axs[0,1].axis('off')
                            axs[0,1].set_title('first  and last ground truth and predicted frame')
                            axs[3,1].clear()
                            axs[3,1].imshow(responses_predicted_full[:,:],aspect='auto',vmin=0,vmax=10)
                            axs[3,1].set_title('predicted responses reconstructed video')
                            axs[0,2].clear()
                            axs[0,2].plot(motion_energy_gt/np.max(motion_energy_gt), color='blue')
                            if i>0:
                                axs[0,2].plot(motion_energy_recon/np.max(motion_energy_recon), color='red')
                            axs[0,2].set_title('motion energy')
                            axs[0,3].clear()
                            axs[0,3].imshow(gradients_fullvid[0,0,eval_frame_skip:].mean(axis=(0)).cpu().detach().numpy(),vmin=-maxgrad/5,vmax=maxgrad/5)
                            axs[0,3].set_title(f'mean grad over space (vmax{np.round(maxgrad/2,4)})')
                            axs[1,2].clear()
                            axs[1,2].plot(video_itter, loss_all)
                            axs[1,2].set_title('response loss')
                            axs[1,3].clear()
                            axs[1,3].plot(video_itter,response_corr)            
                            axs[1,3].set_title('response corr')
                            axs[2,2].clear()
                            axs[2,2].plot(video_itter, video_corr)
                            axs[2,2].set_title('video corr')
                            axs[2,3].clear()
                            axs[2,3].plot(video_itter, video_RMSE)
                            axs[2,3].set_title('video RMSE')
                                                    
                            fig.savefig(save_path+f'reconstruction_summary_m{mouse_index}_d{rep}s{spatial_length_constant}t{temporal_length_constant}.png')
                    
                    # at the end of the reconstruction loop make a dictionary of all relevant variables
                    end_training_time = time.time()
                    training_time = end_training_time - start_training_time
                    recon_dict = {'epochs': epoch_switch[-1],
                                    'strides': strides_all,
                                    'strides_switch': epoch_switch,
                                    'track_itter': track_itter, 
                                    'plot_itter': plot_itter,
                                    'lr': lr,
                                    'loss_func': loss_func,
                                    'response_dropout_rate': response_dropout_rate,
                                    'mouse_index': mouse_index,
                                    'rep': rep,
                                    'spatial_length_constant': spatial_length_constant,
                                    'temporal_length_constant': temporal_length_constant,
                                    'eval_frame_skip': eval_frame_skip,
                                    'video_length': video_length,
                                    'subbatch_size': subbatch_size,
                                    'subbatch_shift': subbatch_shift,
                                    'model_paths': model_path,
                                    'model_list': model_list,
                                    'data_fold': data_fold,
                                    'save_path': save_path,
                                    'vid_init': vid_init,
                                    'lr_warmup_epochs': lr_warmup_epochs,
                                    'use_adam': use_adam,
                                    'adam_beta1': adam_beta1,
                                    'adam_beta2': adam_beta2,
                                    'adam_eps': adam_eps,
                                    'mask': mask,
                                    'mask_update_th': mask_update_th,
                                    'mask_eval_th': mask_eval_th,
                                    'with_gradnorm': with_gradnorm,
                                    'clip_grad': clip_grad,
                                    'device': device,
                                    'video_gt': ground_truth,
                                    'video_pred': reconstruction,
                                    'behavior': behavior,
                                    'pupil_center': pupil_center,
                                    'responses_predicted_gt': responses_predicted_original,
                                    'responses_predicted_full': responses_predicted_full,
                                    'video_itter': video_itter,
                                    'response_loss_full': loss_all,
                                    'response_corr_full': response_corr,
                                    'video_corr': video_corr,
                                    'video_RMSE': video_RMSE,
                                    'motion_energy_gt': motion_energy_gt,
                                    'motion_energy_recon': motion_energy_recon,
                                    'training_time': training_time}
                    # save dictionary
                    np.save(save_path+f'reconstruction_summary_m{mouse_index}_d{rep}s{spatial_length_constant}t{temporal_length_constant}.npy',recon_dict)   
                    
                    # move to next trial
                    plt.close(fig)
