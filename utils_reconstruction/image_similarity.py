import numpy as np
import matplotlib.pyplot as plt

def reconstruction_video_corr(ground_truth, reconstruction, mask):
    # ground_truth: (time, height, width)
    # reconstruction: (time, height, width)
    # mask: (height, width) or (time, height, width)
    # returns: (time, height, width)
    
    if len(mask.shape) == 2:
        mask = mask[None, :, :].repeat(ground_truth.shape[0], axis=0)
    idx_inmask = mask.flatten() == 1
    return np.corrcoef(ground_truth.flatten()[idx_inmask], reconstruction.flatten()[idx_inmask])[0,1]

def reconstruction_video_mean_frame_corr(ground_truth, reconstruction, mask):
    # ground_truth: (time, height, width)
    # reconstruction: (time, height, width)
    # mask: (height, width) or (time, height, width)
    # returns: (time, height, width)
    
    if len(mask.shape) == 3:
        mask = mask[0, :, :]
    idx_inmask = mask.flatten() == 1
    corr_all=[]
    for frame in range(ground_truth.shape[0]):
            corr_all.append(np.corrcoef(ground_truth[frame].flatten()[idx_inmask], reconstruction[frame].flatten()[idx_inmask])[0,1])
    corr_all = np.array(corr_all)
    return np.nanmean(corr_all)#,corr_all

def reconstruction_video_RMSE(ground_truth, reconstruction, mask):
    # ground_truth: (time, height, width)
    # reconstruction: (time, height, width)
    # mask: (height, width) or (time, height, width)
    # returns: (time, height, width)
    # root mean squared error (RMSE)
    
    if len(mask.shape) == 2:
        mask = mask[None, :, :].repeat(ground_truth.shape[0], axis=0)
    idx_inmask = mask.flatten() == 1
    return np.sqrt(np.mean((ground_truth.flatten()[idx_inmask] - reconstruction.flatten()[idx_inmask])**2))

def reconstruction_video_PSNR(ground_truth, reconstruction, mask, max_value=255):
    # ground_truth: (time, height, width)
    # reconstruction: (time, height, width)
    # mask: (height, width) or (time, height, width)
    # returns: (time, height, width)
    # peak signal-to-noise ratio (PSNR)
    
    if len(mask.shape) == 2:
        mask = mask[None, :, :].repeat(ground_truth.shape[0], axis=0)
    idx_inmask = mask.flatten() == 1

    return 10*np.log10(max_value**2/np.mean((ground_truth.flatten()[idx_inmask] - reconstruction.flatten()[idx_inmask])**2))


def reconstruction_video_SSIM(ground_truth, reconstruction, mask, max_value=255):
    # ground_truth: (time, height, width)
    # reconstruction: (time, height, width)
    # mask: (height, width) or (time, height, width)
    # returns: (time, height, width)
    
    # structural similarity index measure (SSIM)
    
    if len(mask.shape) == 2:
        mask = mask[None, :, :].repeat(ground_truth.shape[0], axis=0)
    idx_inmask = mask.flatten() == 1

    k1=1
    k2=1
    c1=(k1*max_value)**2
    c2=(k2*max_value)**2
    c3=c2/2
    alpha = 1
    beta = 1
    gamma = 1    
    means = [np.mean(ground_truth.flatten()[idx_inmask]), np.mean(reconstruction.flatten()[idx_inmask])]
    stds = [np.std(ground_truth.flatten()[idx_inmask]), np.std(reconstruction.flatten()[idx_inmask])]
    covs = np.cov(ground_truth.flatten()[idx_inmask], reconstruction.flatten()[idx_inmask])[0,1]
    luminance = (2*means[0]*means[1] + c1)/(means[0]**2 + means[1]**2 + c1)
    contrast = (2*stds[0]*stds[1] + c2)/(stds[0]**2 + stds[1]**2 + c2)
    structure = (covs + c3)/(stds[0]*stds[1] + c3)
    SSIM = luminance**alpha * contrast**beta * structure**gamma

    return SSIM
    
def video_energy(video, mask):
    # ground_truth: (time, height, width)
    # mask: (height, width) or (time, height, width)
    # returns: (time, height, width)
    
    if len(mask.shape) == 3:
        mask = mask[0, :, :] # make 2D
    idx_inmask = mask.flatten() == 1
    video = video.reshape(video.shape[0],-1) # flatten spatial dimensions
    video = video[:,idx_inmask] # mask spatial dimensions
    frame_energy = np.mean(np.abs(np.diff(video, axis=0)),axis=(1)) # sum of absolute differences (motion energy)
    
    return frame_energy


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