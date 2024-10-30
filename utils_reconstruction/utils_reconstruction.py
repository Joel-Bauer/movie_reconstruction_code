from src.argus_models import MouseModel

import tifffile
import numpy as np
import cv2
import torch 
from torchvision.transforms.functional import gaussian_blur
from src.indexes import IndexesGenerator
from src import constants
from src.inputs import get_inputs_processor
from src.responses import get_responses_processor
from scipy.ndimage.filters import gaussian_laplace
import torch

class Predictor_JB(torch.nn.Module):
    def __init__(self, model, mouse_index, withgrads=False, dummy=True):
        super().__init__()
        self.dummy = dummy
        self.model = model.nn_module # ideally use this (currently i run out of memory)
        self.withgrads = withgrads               
        
        self.mouse_index = mouse_index
        self.frame_stack_size = model.params["frame_stack"]["size"]
        self.frame_stack_step = model.params["frame_stack"]["step"]
        self.indexes_generator = IndexesGenerator(self.frame_stack_size, self.frame_stack_step)
        self.blend_weights = get_blend_weights("ones", self.frame_stack_size) 
        self.blend_weights = torch.tensor(self.blend_weights).to(model.device).requires_grad_(True)
        
        # these are dummy layers to test if autograd works on the local gpu (full model is too big for my gpu)
        self.fakelayer1 = torch.nn.Linear(5*16*64*64, 10)
        self.fakelayer2 = torch.nn.Linear(10, 7939*16)
        self.fakelayer = torch.nn.Sequential(self.fakelayer1, self.fakelayer2).to(model.device).requires_grad_(False)


    
    def forward(self,inputs, with_mask=False, background_video=None):       
        neuron_n = constants.num_neurons[self.mouse_index]
        vid_length = inputs.shape[2]
        responses_predicted = torch.zeros((neuron_n, vid_length)).to(inputs.device)
        blend_weights_all = torch.zeros(vid_length).to(inputs.device)
        
        if with_mask: # whenever input channels = 6 (video(1), mask(1), behaviour(4))
            video = inputs[:,0:1,:,:,:]
            if background_video is None:
                # make a random background with gaussian noise in value range of video and add it to the masked part of video
                background_video = video
                # apply flip h, flip w, flip time
                background_video = torch.flip(background_video, [3])
                background_video = torch.flip(background_video, [4])
                background_video = torch.flip(background_video, [2])
            else:
                background_video = background_video.to(inputs.device)
            mask = torch.clip(inputs[:,1:2,:,:,:],0,1)
            video_new = video*mask + background_video*(1-mask) # alpha blend noise and video with mask
            inputs = torch.cat((video_new, inputs[:,2:,:,:]), axis=1) # recombine but remove mask
            
        for index in range(self.indexes_generator.behind, vid_length - self.indexes_generator.ahead):
            # get frame indexes
            indexes = self.indexes_generator.make_indexes(index)
            
            # get prediction
            if self.dummy: #### to debug just without using the neural encoder model
                temp = inputs[:,indexes].flatten(start_dim=0)
                prediction = self.fakelayer(temp).reshape((neuron_n, len(indexes)))
            else:
                temp = inputs[:,:,indexes]
                prediction = self.model(temp, self.mouse_index) 
                
            # add to responses_predicted
            responses_predicted[..., indexes] += prediction[0]

            # now the same with blend_weights
            blend_weights_all[indexes] += self.blend_weights

        responses_predicted = responses_predicted/torch.clip(blend_weights_all, 1.0, None)
        return responses_predicted

# calcualte stride and epoch 
def stride_calculator(minibatch=32,n_steps=3,epoch_number_first=1000,kernel_size=32,minimum_stride=2, epoch_reducer=0.5):
    stride_start = minibatch
    stride_end = minibatch-kernel_size
    if stride_end < minimum_stride:
        stride_end = minimum_stride
    strides = np.linspace(stride_start,stride_end,n_steps).astype(int) 
    # i need a sequence that starts with the first epoch number and decreases by half each time
    epochs = [epoch_number_first]
    for i in range(1,n_steps):
        epochs.append(round(epochs[i-1]*epoch_reducer))
    epochs = np.array(epochs)
    epoch_switch = np.cumsum(epochs)
    
    return strides, epoch_switch

def load_trial_data(trial_data_all, model, trial, skip_frames, length=None, print_dims=True):
    trial_data = trial_data_all[trial] # try 42 from fold 0 as this works well
    
    if length is None:
        length = trial_data["length"] 
    
    video = np.load(trial_data["video_path"])[..., skip_frames:skip_frames+length] # this has form hight, width, time
    behavior = np.load(trial_data["behavior_path"])[..., skip_frames:skip_frames+length]
    pupil_center = np.load(trial_data["pupil_center_path"])[..., skip_frames:skip_frames+length]
    responses = np.load(trial_data["response_path"])[..., skip_frames:skip_frames+length]
    if print_dims:
        print(f"video.shape: {video.shape}" '\n'
            f"behavior.shape: {behavior.shape}" '\n'
            f"pupil_center.shape: {pupil_center.shape}"'\n'
            f"responses.shape: {responses.shape}" '\n')

    inputs_processor = get_inputs_processor(*model.params["inputs_processor"])
    inputs = inputs_processor(video, behavior, pupil_center)
    
    if print_dims:
        print(f"inputs.shape: {inputs.shape}")
    responses_processor = get_responses_processor(*model.params["responses_processor"])
    responses = responses_processor(responses)
    
    return inputs, responses, video, behavior, pupil_center

# def loss functions
def response_loss_function(responses_predicted,responses,loss_func='poisson',dropout_rate=0,drop_method='zero_pred',mask=None):
    # add dropout to responses and responses_predicted
    if dropout_rate>0:
        # for regularization dropout mask is None, for testing the effect of reduced population size mask is not None
        if mask is not None:
            # Ensure the tensors are on the same device
            device = responses_predicted.device
            mask = torch.rand((responses_predicted.shape[0],1), device=device) > dropout_rate # above == keep, so dropout_rate=0.1 means 10% dropout
            mask = mask.repeat(1,responses_predicted.shape[1])
        
        #if drop_method == 'zero_pred':
        if drop_method == 'zero_pred':
            responses_predicted = responses_predicted * mask
        elif drop_method == 'zero_pred_n_true':
            responses_predicted = responses_predicted * mask
            responses = responses * mask
        elif drop_method == 'set_pred_to_true':
            #instead of make 0 i want to switch out the prediction for the true response 
            responses_predicted = responses_predicted * mask + responses * (1-mask)
    
    if loss_func == 'mse':
        loss = torch.nn.MSELoss()
    elif loss_func == 'poisson':
        loss = torch.nn.PoissonNLLLoss(log_input=False, full=False)
    return loss(responses_predicted.float(),responses.float())

def init_weights(inputs,vid_init,device):
    # initialize video
    if vid_init == 'noise':
        video_pred = torch.randn((1,1,inputs.shape[1],inputs.shape[2],inputs.shape[3])).to(device)
        video_pred = min_max_rescale(video_pred, min=0, max=255)
        
    elif vid_init == 'static_noise':
        video_pred = torch.randn((1,1,1,inputs.shape[2],inputs.shape[3])).to(device)
        video_pred = video_pred.repeat(1,1,inputs.shape[1],1,1)
        video_pred = min_max_rescale(video_pred, min=0, max=255)
        
    elif vid_init == 'gt_vid_first_1':
        video_pred = inputs[0:1,0:1,:,:].repeat(1,1,inputs.shape[1],1,1)
        video_pred = min_max_rescale(video_pred, min=0, max=255)
        
    elif vid_init == 'gt_vid':
        video_pred = inputs[None, 0:1,:,:,:]
        video_pred = min_max_rescale(video_pred, min=0, max=255)
        
    elif vid_init == 'gray':
        video_pred = torch.ones((1,1,inputs.shape[1],inputs.shape[2],inputs.shape[3])).to(device)
        video_pred = video_pred*(255/2)
        
    else:
        print('no valid video initialization chosen')
        exit()
    video_pred.to(device)

    return video_pred

# process batch to match input dims
def get_blend_weights(name: str, size: int):
    if name == "ones":
        return np.ones(size)
    elif name == "linear":
        return np.linspace(0, 1, num=size)
    else:
        raise ValueError(f"Blend weights '{name}' is not supported")
        
def min_max_rescale(data, min=0, max=1):
    """
    Rescale the input data to the specified range.
    
    Parameters:
    - data: Input data as a NumPy array or PyTorch tensor.
    - new_min: The minimum value of the output range.
    - new_max: The maximum value of the output range.
    
    Returns:
    - The rescaled data.
    """
    if isinstance(data, np.ndarray):
        # For NumPy arrays
        min_val = np.min(data).flatten()
        max_val = np.max(data).flatten()
        if max_val-min_val == 0:
            rescaled_data = np.ones_like(data)*(max_val-min_val)/2 + min_val
        else:
            rescaled_data = (data - min_val) / (max_val - min_val) * (max - min) + min
    elif isinstance(data, torch.Tensor):
        # For PyTorch tensors
        min_val = torch.min(data).flatten()
        max_val = torch.max(data).flatten()
        if max_val-min_val == 0:
            rescaled_data = torch.ones_like(data)*(max_val-min_val)/2 + min_val
        else:
            rescaled_data = (data - min_val) / (max_val - min_val) * (max - min) + min
    else:
        raise ValueError("Input data must be a NumPy array or a PyTorch tensor.")
    
    return rescaled_data

def cat_video_behaviour(video_input, behaviour_input):
    if behaviour_input.shape[1] == 1: # duplicate to match length of vid
        behaviour_input = behaviour_input.repeat(1,video_input.shape[1],1,1)
    return torch.cat((video_input, behaviour_input), axis=1)

def cat_video_behaviour_mask(video_input, mask, behaviour_input):
    return torch.cat((video_input, mask, behaviour_input), axis=1)

def save_avi(ground_truth, reconstructed, filename):
    ground_truth = min_max_rescale(np.moveaxis(ground_truth,[2],[0]), min=0, max=255)
    reconstructed = min_max_rescale(reconstructed.cpu().detach(), min=0, max=255)
    reconstructed = reconstructed[:,14:14+36,:]
    
    output_video = np.concatenate((ground_truth,reconstructed),axis=1).astype('uint8') # cat GT and predicted move in hight dimension
    
    # compression look shit... use something else!
    forcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, forcc, 30.0, (64,72),0)
    for frame in output_video:
        #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()

    return output_video


# function to save video as tif
def save_tif(ground_truth, reconstructed, filename, mask=None):
    if mask is not None:
        # mask = mask.repeat(ground_truth.shape[0],1,1).cpu().detach().numpy()
        mask = np.tile(mask,(ground_truth.shape[0],1,1))
        ground_truth = ground_truth*mask + np.ones_like(ground_truth)*(1-mask)*255/2 # alpha blend gray screen and video with mask
        reconstructed = reconstructed*mask + np.ones_like(reconstructed)*(1-mask)*255/2 # alpha blend gray screen and video with mask

    output_video = np.concatenate((ground_truth,reconstructed),axis=1).astype('uint8') # cat GT and predicted move in hight dimension
    
    # tifffile.imsave(filename, output_video)
    tifffile.imwrite(filename, 
                    output_video,
                    imagej=True,
                    metadata = {'unit': 'um','fps': 30.0,'axes': 'TYX',})
    return output_video




## currently not used
class LaplaceLoss3D(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(LaplaceLoss3D, self).__init__()
        # sigma to torch tensor
        self.device = device

    def laplacian_kernel_2d(self) -> torch.Tensor:
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float)
        # kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float)
        return kernel
    
    def laplacian_kernel_1d(self) -> torch.Tensor:
        kernel = torch.tensor([-1, 2, -1], dtype=torch.float)
        return kernel

    def laplacian_space_filter(self, video: torch.Tensor) -> torch.Tensor:
        laplacian2d_kernel = self.laplacian_kernel_2d()
        # Apply Laplacian filter
        video = torch.nn.functional.conv3d(video, weight=laplacian2d_kernel.view(1, 1, 1, 3, 3).to(self.device), padding=1)        
        return video.squeeze_(0) # Make 4D again

    def laplacian_time_filter(self, video: torch.Tensor) -> torch.Tensor:
        laplacian1d_kernel = self.laplacian_kernel_1d()
        # Apply Laplacian filter
        video = torch.nn.functional.conv3d(video, weight=laplacian1d_kernel.view(1, 1, 3, 1, 1).to(self.device), padding=1)
        return video.squeeze_(0) # Make 4D again

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        space_filtered_video = self.laplacian_space_filter(video)
        time_filtered_video = self.laplacian_time_filter(video)
        
        return torch.mean(space_filtered_video)*video.shape[-1] + torch.mean(time_filtered_video)
    
class GaussianLaplaceLoss3D(torch.nn.Module):
    def __init__(self, sigma, device='cuda'):
        super(GaussianLaplaceLoss3D, self).__init__()
        # sigma to torch tensor
        self.sigma = torch.tensor(sigma, dtype=torch.float)
        self.device = device

    def gaussian_kernel_1d(self, sigma, num_sigmas = 3.) -> torch.Tensor:
        radius = torch.ceil(num_sigmas * sigma).item()
        support = torch.arange(-radius, radius + 1, dtype=torch.float)
        kernel = torch.exp(-0.5 * (support / sigma) ** 2)
        return kernel / kernel.sum()

    def laplacian_kernel_2d(self) -> torch.Tensor:
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float)
        return kernel
    
    def laplacian_kernel_1d(self) -> torch.Tensor:
        kernel = torch.tensor([-1, 4, -1], dtype=torch.float)
        return kernel

    def gaussian_laplacian_filter_3d(self, video: torch.Tensor) -> torch.Tensor:
        gaussian_kernel = self.gaussian_kernel_1d(self.sigma)
        laplacian2d_kernel = self.laplacian_kernel_2d()
        laplacian1d_kernel = self.laplacian_kernel_1d()
        
        padding = len(gaussian_kernel) // 2
        
        # Apply Gaussian filter
        video = torch.nn.functional.conv3d(video, weight=gaussian_kernel.view(1, 1, -1, 1, 1).to(self.device), padding=(padding, 0, 0))
        video = torch.nn.functional.conv3d(video, weight=gaussian_kernel.view(1, 1, 1, -1, 1).to(self.device), padding=(0, padding, 0))
        video = torch.nn.functional.conv3d(video, weight=gaussian_kernel.view(1, 1, 1, 1, -1).to(self.device), padding=(0, 0, padding))
        
        # Apply Laplacian filter
        video = torch.nn.functional.conv3d(video, weight=laplacian2d_kernel.view(1, 1, 1, 3, 3).to(self.device), padding=1)
        video = torch.nn.functional.conv3d(video, weight=laplacian1d_kernel.view(1, 1, 3, 1, 1).to(self.device), padding=1)
        
        return video.squeeze_(0) # Make 4D again

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        filtered_video = self.gaussian_laplacian_filter_3d(video)
        return torch.mean(filtered_video)
    
def apply_gaussian_blur_3d(input_tensor, sigma_time, sigma_spatial, factor=0.001):
    # Separate the time dimension
    channels, time, height, width = input_tensor.shape
    
    # Apply Gaussian blur to spatial dimensions
    kernel_size_spatial = int(3 * sigma_spatial)
    if kernel_size_spatial % 2 == 0:
        kernel_size_spatial += 1 # Ensure odd kernel size
    blurred_spatial = gaussian_blur(input_tensor, kernel_size_spatial, sigma_spatial)
    
    if sigma_time > 0:
        # Apply Gaussian blur to time dimension
        # This is a simplified example. You might need a more sophisticated approach for the time dimension.
        kernel_size_time = int(3 * sigma_time)
        if kernel_size_time % 2 == 0:
            kernel_size_time += 1 # Ensure odd kernel size
        blurred_time = torch.zeros_like(input_tensor)
        for w in range(width):
            blurred_time[:, :, :, w] = gaussian_blur(input_tensor[:, :, :, w], [kernel_size_time, 1], [sigma_time,1])
        # Combine the results
        blurred_tensor =  blurred_time + blurred_spatial
    else:
        blurred_tensor = blurred_spatial
    
    if factor<=0: # just a hack if factor is set to 0 or negative then its the original input that is scaled 
        blurred_tensor = blurred_tensor + input_tensor*np.abs(factor)
    else:
        blurred_tensor = factor*blurred_tensor + input_tensor
    return blurred_tensor

def luminance_contrast_match(video_gt,video_recon,mask,mask_th): # assumes single movie as input
    # video_gt: (time, height, width)
    # video_recon: (time, height, width)
    # mask: (height, width)
    # mask_th: threshold for mask
    # returns: (time, height, width)
    expanded_mask = np.repeat(np.where(mask>= mask_th,1,np.nan),video_gt.shape[0],axis=0)
    video_gt_masked = video_gt*expanded_mask
    video_gt_masked_mean = np.nanmean(video_gt_masked, axis=(-1,-2,-3))
    video_gt_masked_std = np.nanstd(video_gt_masked, axis=(-1,-2,-3))
    
    video_recon_masked = video_recon*expanded_mask
    video_recon_masked_mean = np.nanmean(video_recon_masked, axis=(-1,-2,-3))
    video_recon_masked_std = np.nanstd(video_recon_masked, axis=(-1,-2,-3))
    
    video_recon_masked_zscored = (video_recon_masked - video_recon_masked_mean) / video_recon_masked_std
    video_recon_masked_norm = (video_recon_masked_zscored*video_gt_masked_std) + video_gt_masked_mean
    video_recon_masked_norm = np.clip(video_recon_masked_norm, 0, 255)
    
    video_recon_masked_norm = video_recon_masked_norm*expanded_mask
    video_recon_masked_norm[np.isnan(video_recon_masked_norm)] = 255/2
    
    video_gt_masked[np.isnan(video_gt_masked)] = 255/2
    
    return video_recon_masked_norm, video_gt_masked