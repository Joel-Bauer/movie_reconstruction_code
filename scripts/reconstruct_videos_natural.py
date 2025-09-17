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
model_list = [0] # 0,1,2,3,4,5,6 # if this is more than one then the gradient is averaged across models at each step
number_models = np.array(model_list).shape[0]
animals = [0,1,2,3,4] # range(0,5) 
start_trial = [0]
end_trial = [10] # not incluing this one
random_trials = False # if true then random trials are chosen, default False
video_length = None # None. max is 300 but can be too much for your gpu
load_skip_frames = 0 # in case beginning of video should be skipped, default 0
mask_update_th = 0.5 # [0:1], defaul 0.5
mask_eval_th = 1 # [0:1], default 1
no_train_mask = False
strict_mask = False # if true then the mask is only applied to the video channel, otherwise it is applied to all channels
optimize_given_predicted_responese = False # to optimize to the predicted responses rather than the gt responses


# Reduced population size 
population_reduction = 0 # a fraction (0-1) of the population to drop, new sample for each trial (1/2,3/4,7/8)

# option to control parameters as inputs
parser = argparse.ArgumentParser(description='optional parameters as inputs. mouse,model_list')
parser.add_argument('--model_list', type=int, nargs='+', default=model_list, help='list of models to use for reconstruction 0 to 6')
parser.add_argument('--animals', type=int, nargs='+', default=animals, help='list of animals to reconstruct videos from [0,1,2,3,4]')
parser.add_argument('--start_trial', type=int, nargs='+', default=start_trial, help='first trail to reconstruction 0')
parser.add_argument('--end_trial', type=int, nargs='+', default=end_trial, help='last trial to reconstruct (not including this one) 10')
parser.add_argument('--population_reduction', type=float, default=population_reduction, help='fraction of population to reduce 0,0.5,0.75,0.875')
parser.add_argument('--no_train_mask', action='store_true', help='if true then the mask is not applied to the video channel, but to all channels')
parser.add_argument('--strict_mask', action='store_true', help='if true then the mask is only applied to the video channel, otherwise it is applied to all channels')
parser.add_argument('--optimize_given_predicted_responese', action='store_true', help='if true then the optimization is done to the predicted responses rather than the gt responses')
args = parser.parse_args()
model_list = np.array(args.model_list) 
number_models = np.array(model_list).shape[0]
animals = np.array(args.animals)
start_trial = args.start_trial[0]
end_trial = args.end_trial[0]
population_reduction = args.population_reduction
print('population_reduction: ' + str(population_reduction))
no_train_mask = args.no_train_mask
strict_mask = args.strict_mask
optimize_given_predicted_responese = args.optimize_given_predicted_responese



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
save_path_sufix = 'hpc_round11'
if optimize_given_predicted_responese:
    save_path_sufix = save_path_sufix + '_fit_to_predicted'
if population_reduction > 0:
    save_path_sufix = save_path_sufix + f'_popreduc'
if randomize_neurons:
    save_path_sufix = save_path_sufix + '_randneurons'
if no_train_mask:
    mask_update_th = 0
    save_path_sufix = save_path_sufix + '_nontrainmask'
if loss_func == 'mse':
    save_path_sufix = save_path_sufix + '_mse'
elif strict_mask:
    mask_update_th = 1
    save_path_sufix = save_path_sufix + '_strictmask'

save_path = f'reconstructions/modelfold{model_list}_data{data_fold}_pop{round(100-population_reduction*100)}_{save_path_sufix}/'

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

# create all predictors
if optimize_given_predicted_responese:
    # then all the predictors need to be loaded
    predictor = [None]*len(model_path)
    for n in range(0,len(predictor)):
        model_path_temp = model_path[n]
        print('predicting true resp with model: ', model_path_temp)
        predictor[n] = Predictor(model_path=model_path_temp, device=device, blend_weights="ones")
else: 
    # then only the models used for reconstuction are loaded
    predictor = [None]*number_models
    for n in range(0,len(predictor)):
        model_path_temp = model_path[model_list[n]]
        print('predicting true resp with model: ', model_path_temp)
        predictor[n] = Predictor(model_path=model_path_temp, device=device, blend_weights="ones")        
        

## loop through trials
print('\nget a batch')
for mouse_index in animals:
    # get data
    mouse_index = mouse_index
    mouse = constants.index2mouse[mouse_index]
    mouse_data = get_mouse_data(mouse=mouse, splits=[data_fold])
    trial_data_all=mouse_data['trials']    
    print('total trials: ' + str(len(trial_data_all)))
    if end_trial is None:
        end_trial = len(trial_data_all)
    
    # load mask
    mask = np.load(f'reconstructions/masks/mask_m{mouse_index}.npy')
    mask_update = torch.tensor(np.where(mask >= mask_update_th,1,0)).to(device)
    mask_eval = torch.tensor(np.where(mask >= mask_eval_th,1,0)).to(device)
    print('mask shape: ', mask.shape)
    
    for trial_n in range(start_trial,end_trial):
        # get trial data path
        torch.cuda.empty_cache() 
        if random_trials==True:
            trial = np.random.randint(0,len(trial_data_all)-1, 1)[0]
        else:
            trial = trial_n
        trial_data = trial_data_all[trial] 
        print('trial paths:')
        print(trial_data)

        # load trial data
        if video_length is None:
            video_length = trial_data["length"] 
        inputs, responses, video, behavior, pupil_center = utils.load_trial_data(trial_data_all, model[0], trial, load_skip_frames, length=video_length)
        inputs=inputs.to(device)
        responses=responses.to(device)
        
        # randomize response order with specified seed
        if randomize_neurons:
            manual_seed = (trial_n+1)*(mouse_index+1) # this means there will be a different seed for most trials and mice but conistent across model rusn
            print(f'random seed for randomizing neurons: {manual_seed}')
            torch.manual_seed(manual_seed) # manual seed so each video has the same shuffled neuron order across models
            responses = responses[torch.randperm(responses.shape[0]),:]
        
        # reduce population (optional)
        if population_reduction > 0:
            manual_seed = (trial_n+1)*(mouse_index+1) # this means there will be a different seed for most trials and mice but conistent across model runs
            print(f'random seed for population dropout: {manual_seed}')
            torch.manual_seed(manual_seed) # manual seed so each video has the same neurons across models
            population_mask = torch.rand((responses.shape[0],1), device=device) > population_reduction  
            print(f'population mask shape: {population_mask.shape}')
            print(f'repponses shape: {responses.shape}')
            responses = responses * population_mask.repeat(1,responses.shape[1])
        else:
            population_mask = None        
        
        # prepare mask
        mask_update_expanded = mask_update.repeat(1,1,inputs.shape[1],1,1)
        mask_eval_expanded = mask_eval.repeat(1,1,inputs.shape[1],1,1)

        # prediction based on full video
        responses_predicted_original = np.zeros((constants.num_neurons[mouse_index], video_length, len(predictor)))
        for n in range(0,len(predictor)):
            responses_predicted_original[:,:,n] = predictor[n].predict_trial(
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index)
            
        # if performing a population ablation experiment, ablate neurons from the population prediction aswell
        if population_mask is not None:
            responses_predicted_original = responses_predicted_original * population_mask[:,None,:].cpu().detach().numpy()

        # get correlation and loss between predicted responses and gt responses
        loss_gt = utils.response_loss_function(torch.from_numpy(responses_predicted_original.mean(axis=2)).to(device),
                                        responses.clone().detach().to(device), mask=population_mask)       
        response_corr_gt = np.corrcoef(responses[:,:].cpu().detach().numpy().flatten(), 
                                    responses_predicted_original.mean(axis=2).flatten())[0,1]
        print( f'gt test loss: {loss_gt.item()}')
        
        # if we want to optimize to the average predicted responses across models rather than the gt responses
        if optimize_given_predicted_responese:
            alternative_response_gt = torch.from_numpy(responses_predicted_original.mean(axis=2).copy()).to(device)
        
        # prediction based on eval masked video 
        responses_predicted_original_masked = np.zeros((constants.num_neurons[mouse_index], video_length, len(predictor)))
        for n in range(0,len(predictor)):
            responses_predicted_original_masked[:,:,n] = predictor[n].predict_trial(
                video=video*np.moveaxis(mask_eval_expanded[0,0,:,14:14+36,:].detach().cpu().numpy(),[0],[-1]),
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index)
            
        # if performing population ablation experiment, ablate neurons from the population prediction aswell
        if population_mask is not None:
            responses_predicted_original_masked = responses_predicted_original_masked * population_mask[:,None,:].cpu().detach().numpy()
            
        # get correlation and loss between predicted responses and gt responses
        loss_gt_masked = utils.response_loss_function(torch.from_numpy(responses_predicted_original_masked.mean(axis=2).copy()).to(device),
                                        responses.clone().detach().to(device), mask=population_mask)   
        response_corr_gt_masked = np.corrcoef(responses[:,:].cpu().detach().numpy().flatten(), 
                                    responses_predicted_original_masked.mean(axis=2).flatten())[0,1]
        print( f'gt test loss with mask: {loss_gt_masked.item()}')

        # response prediction to gray screen
        responses_predicted_gray = np.zeros((constants.num_neurons[mouse_index], video_length, len(predictor)))
        for n in range(0,len(predictor)):
            responses_predicted_gray[:,:,n] = predictor[n].predict_trial(
                video=np.ones_like(video)*(255/2),
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index)
        
        # apply population_mask to response predictions
        if population_mask is not None:
            responses_predicted_gray = responses_predicted_gray * population_mask[:,None,:].cpu().detach().numpy()
        
        # define preditor which tracks gradients
        predictor_withgrads = [None]*number_models 
        for n in range(0,number_models): # only the selected models not all
            predictor_withgrads[n] = utils.Predictor_JB(model[n], mouse_index, withgrads=True, dummy=False)
        print('\nskipping test of gradient tracking')

        # plot the responses and the predicted responses to check
        print('responses range: ' + str(responses.min().cpu().detach().numpy()) + ' to ' + str(responses.max().cpu().detach().numpy()))
        print('vid range: ' + str(inputs[0,:].min().cpu().detach().numpy()) + ' to ' + str(inputs[0,:].max().cpu().detach().numpy()))

        ### export figure containing first frame of video, behaviour, responeses and predictions
        fig, axs = plt.subplots(4,4, figsize=(20, 20))
        fig.suptitle(f'mouse {mouse_index} trial {trial} model {model_list}', fontsize=16)
        axs[0,0].imshow(np.concatenate((video[:,:,0],video[:,:,-1]),axis=1),cmap='gray')
        axs[0,0].axis('off')
        axs[0,0].set_title('video frame 0 and ' + str(video.shape[2]))
        axs[1,0].plot(behavior[:,:].T)
        axs[1,0].set_title('behaviour')
        axs[2,0].plot(pupil_center[:,:].T)
        axs[2,0].set_title('pupil position')
        axs[1,1].imshow(responses[:,:].cpu().detach().numpy(),aspect='auto',vmin=0,vmax=10)
        axs[1,1].set_title('responses')
        axs[2,1].imshow(responses_predicted_original.mean(axis=2),aspect='auto',vmin=0,vmax=10) # predictions based on selected model not all
        axs[2,1].set_title('predicted responses')
        axs[3,0].imshow(responses_predicted_gray.mean(axis=2),aspect='auto',vmin=0,vmax=10) # predictions based on selected model not all
        axs[3,0].set_title('predicted responses gray screen')
        axs[0,2].plot(np.sum(np.abs(np.diff(video, axis=2)),axis=(0,1)))
        axs[0,2].set_title('video motion energy')
        fig.savefig(save_path+f'reconstruction_summary_m{mouse_index}_t{trial}.png')
        
        ### lets get started ###
        # initialize video
        print('\n')
        video_pred = utils.init_weights(inputs,vid_init,device)
        print('video_pred shape: ' + str(video_pred.shape))
        
        ## loop to optimize video
        print('\n')
        
        # prepare image_sim vars
        video_corr = []
        video_RMSE = []
        video_itter = []
        loss_all = []
        response_corr = []
                
        print(f'reconstructing mouse {mouse_index} trial {trial}')
        progress_bar = tqdm(range(epoch_switch[-1]))
        start_training_time = time.time()
        for i in progress_bar:
            for n in range(0,n_steps):
                if i < epoch_switch[n]:
                    subbatch_shift = strides_all[n]
                    break
                            
            number_of_subbatches = 2+(inputs.shape[1]-subbatch_size)//subbatch_shift
                        
            # initialize gradient accumulator
            gradients_fullvid = torch.zeros_like(video_pred).repeat(number_models,number_of_subbatches,1,1,1,1).to(device).requires_grad_(False) # subbatch, channel, time, height, width
            gradients_fullvid = gradients_fullvid.fill_(float('nan'))
            gradnorm = torch.zeros(number_models,number_of_subbatches).to(device)
            for n in range(0,number_models):
                for subbatch in range(0,number_of_subbatches,1):
                    
                    if subbatch == number_of_subbatches-1:
                        start_frame = inputs.shape[1]-subbatch_size
                        end_frame = inputs.shape[1]
                    else:
                        start_frame = subbatch*subbatch_shift
                        end_frame = subbatch*subbatch_shift+subbatch_size
                    subbatch_frames = range(start_frame,end_frame,1)

                    # add input noise (optional)
                    if input_noise>0:
                        added_video_noise = torch.randn_like(video_pred[:,:,subbatch_frames,:,:])*input_noise
                    else:
                        added_video_noise = torch.zeros_like(video_pred[:,:,subbatch_frames,:,:])
                    
                    # concatenate video and behaviour
                    input_prediction = utils.cat_video_behaviour(video_pred[:,:,subbatch_frames,:,:]+added_video_noise,inputs[None,1:,subbatch_frames,:,:]).detach().requires_grad_(True)    
                    
                    # run model
                    responses_predicted_new = predictor_withgrads[n](input_prediction)
                    
                    if optimize_given_predicted_responese:
                        # get a gradients
                        loss = utils.response_loss_function(responses_predicted_new,
                                                        alternative_response_gt[:,subbatch_frames].clone().detach().to(device).requires_grad_(True),
                                                        mask=population_mask)                        
                    else:
                        # get a gradients
                        loss = utils.response_loss_function(responses_predicted_new,
                                                        responses[:,subbatch_frames].clone().detach().to(device).requires_grad_(True),
                                                        mask=population_mask)
                    loss.backward()
                    gradients = input_prediction.grad   
                    
                    # normalize gradients
                    gradnorm[n,subbatch] = torch.norm(gradients)
                    if with_gradnorm:
                        gradients = gradients / (gradnorm[n,subbatch] + 1e-6)
                    else:
                        gradients = gradients*100
                    
                    # add gradients of video channel to accumulator
                    gradients_fullvid[n,subbatch,:,subbatch_frames,:,:] = gradients[:,0:1,:,:,:]

            # average gradients across subbatches and model pedictions excluding 0s
            gradients_fullvid = torch.nanmean(gradients_fullvid, axis=1, keepdim=True) # mean across all subbatches
            
            # average gradients across models (in case of graident ensembling)
            gradients_fullvid = torch.nanmean(gradients_fullvid, axis=0, keepdim=False) # mean across all models
                                    
            # mask gradients
            gradients_fullvid = gradients_fullvid*mask_update_expanded
            
            # clip gradients
            if clip_grad is not None:
                gradients_fullvid = torch.clip(gradients_fullvid, -1*clip_grad, clip_grad)

            # GD or Adam based weight update
            if i > 0: # to track where we start from we just skip the first update
                if lr_warmup_epochs>0 and i<lr_warmup_epochs: # warup period
                    lr_current = lr*min(1,i/lr_warmup_epochs)
                else: # otherwise use set lr
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
            
            # pixel decay (not used in the paper)
            if pix_decay_rate>0:
                video_pred = ((video_pred-255/2)*(1-pix_decay_rate))+(255/2)
            
            # reset gradients
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
                responses_predicted_full = np.zeros((constants.num_neurons[mouse_index], video_length,len(predictor)))
                for n in range(0,len(predictor)):
                    responses_predicted_full[:,:,n] = predictor[n].predict_trial(
                        video=np.moveaxis(reconstruction_masked,[0],[2]),
                        behavior=behavior,
                        pupil_center=pupil_center,
                        mouse_index=mouse_index)
                responses_predicted_full = responses_predicted_full.mean(axis=2) # here we always average over all selected models
                
                # apply population_mask to response predictions
                if population_mask is not None:
                    responses_predicted_full = responses_predicted_full * population_mask.cpu().detach().numpy() 
                    
                loss_full = utils.response_loss_function(torch.from_numpy(responses_predicted_full[:,eval_frame_skip:]).to(device),
                                                    responses[:,eval_frame_skip:].clone().detach().to(device), mask=population_mask)   
                loss_all.append(loss_full.item())
                
                # save movie as tiff
                concat_video = utils.save_tif(ground_truth, reconstruction, save_path+f'optimized_input_m{mouse_index}_t{trial}.tif',mask=mask_cropped)
                                                
                # calculate similarity metrics
                video_corr.append(im_sim.reconstruction_video_corr(ground_truth[eval_frame_skip:], reconstruction[eval_frame_skip:], mask_cropped))
                video_RMSE.append(im_sim.reconstruction_video_RMSE(ground_truth[eval_frame_skip:], reconstruction[eval_frame_skip:], mask_cropped))
                video_itter.append(i)    
                response_corr.append(np.corrcoef(responses[:,:].cpu().detach().numpy().flatten(), 
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
                axs[1,2].axhline(y=loss_gt.item(), color='k', linestyle='--')
                axs[1,2].axhline(y=loss_gt_masked.item(), color='b', linestyle='--')
                axs[1,2].set_title('response loss')
                axs[1,3].clear()
                axs[1,3].plot(video_itter,response_corr)
                axs[1,3].axhline(y=response_corr_gt.item(), color='k', linestyle='--')                
                axs[1,3].axhline(y=response_corr_gt_masked.item(), color='b', linestyle='--')                
                axs[1,3].set_title('response corr')
                axs[2,2].clear()
                axs[2,2].plot(video_itter, video_corr)
                axs[2,2].set_title('video corr')
                axs[2,3].clear()
                axs[2,3].plot(video_itter, video_RMSE)
                                        
                fig.savefig(save_path + f'reconstruction_summary_m{mouse_index}_t{trial}.png')
        
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
                        'trial': trial,
                        'load_skip_frames': load_skip_frames,
                        'eval_frame_skip': eval_frame_skip,
                        'video_length': video_length,
                        'subbatch_size': subbatch_size,
                        'subbatch_shift': subbatch_shift,
                        'model_paths': model_path,
                        'model_list': model_list,
                        'data_fold': data_fold,
                        'save_path': save_path,
                        'population_reduction': population_reduction,
                        'population_mask': population_mask,
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
                        'responses': responses,
                        'responses_predicted_gt': responses_predicted_original, # one for each model. shape: neurons, time, models
                        'responses_predicted_full': responses_predicted_full, # one for model ensemble. shape: neurons, time 
                        'video_itter': video_itter,
                        'response_loss_gt': loss_gt,
                        'response_loss_full': loss_all,
                        'response_corr_gt': response_corr_gt,
                        'response_corr_full': response_corr,
                        'video_corr': video_corr,
                        'video_RMSE': video_RMSE,
                        'motion_energy_gt': motion_energy_gt,
                        'motion_energy_recon': motion_energy_recon,
                        'training_time': training_time}
        # save dictionary
        np.save(save_path+f'reconstruction_summary_m{mouse_index}_t{trial}.npy',recon_dict)   
         
        # move to next trial
        plt.close(fig)
