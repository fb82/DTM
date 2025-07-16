import sys
import os
import torch
import cv2
import poselib
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_kpts
import numpy as np
import src.dtm as dtm
import hz.hz as hz
import matplotlib.pyplot as plt
import warnings

if __name__ == "__main__":
    # device to use (with the exception of Blob Matching running always on CPU to avoid OOM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if you need, change to
    # device = 'cpu'
   
    warnings.warn("if your GPU has little amount of memory (i.e. 4GB), set device='cpu' or change the image pair as detailed in this demo.py code")    
    
    im_pair = ['data/DC/dc0.png', 'data/DC/dc2.png']
    # or if you have low amount of GPU memory
    # im_pair = ['data/ET/et000.jpg', 'data/ET/et001.jpg']

    # you can give non-default input image pair as further arguments 
    if len(sys.argv) >= 3: img = [sys.argv[1], sys.argv[2]]
    else: img = im_pair

    # more keypoint detectors (or just one if you don't have enough memory)
    detectors = {'Hz+', 'DoG'}
    
    # patch preprocessing
    patchers = {'AffNet', 'OriNet'}
    
    # matching
    matcher = {'Blob Matching'}
    # or just
    # matcher = {'Mutual Nearest Neighbor (MNN)': 0.9}
    
    # in case one does not want to use dissimilarity values of matches but only the spatial keypoint localization on the images
    dtm_only_spatial = False
    
    # Delaunay pre-quantization to redeuce the spatial grid resolution
    # kp = (round(kp * s + t) - t) / s
    dtm_st = [1., 0.] 
    
    # visualize DTM steps
    dtm_show_in_progress = False

    # DTM border handling
    # the original Matlab approach is not implemented due to laziness (the matlab boundary function does not exist in Python)
    # but this should work similar
    dtm_prepare_data = dtm.prepare_data_shaped         

    # RANSAC fundamental matrix estimation parameters    
    poselib_params = {            
     'max_iterations': 100000,
     'min_iterations': 50,
     'success_prob': 0.9999,
     'max_epipolar_error': 3,
     }    
    
    # guided matching iterations, done by forcing their similarity values to the lowest value    
    # currently not done
    ii = 0
    # for doing once set to 
    # ii = 1
    
    if (not os.path.isfile(img[0])) or (not os.path.isfile(img[1])):
        print('one or both input images not found!')
    
    with torch.no_grad():        
        laf0 = torch.zeros((1, 0, 2, 3), device=device, dtype=torch.float)        
        laf1 = torch.zeros((1, 0, 2, 3), device=device, dtype=torch.float)        

        # Hz+
        if 'Hz+' in detectors:
            hz0, _ = hz.hz_plus(hz.load_to_tensor(img[0]).to(torch.float), output_format='laf')
            hz0 = KF.ellipse_to_laf(hz0[None]).to(device).to(torch.float)
            laf0 = torch.concat((laf0, hz0), dim=1)

            hz1, _ = hz.hz_plus(hz.load_to_tensor(img[1]).to(torch.float), output_format='laf')
            hz1 = KF.ellipse_to_laf(hz1[None]).to(device).to(torch.float)
            laf1 = torch.concat((laf1, hz1), dim=1)

        # DoG
        if 'DoG' in detectors:        
            dog = cv2.SIFT_create(nfeatures=8000, contrastThreshold=-10000, edgeThreshold=10000)

            dog0 = laf_from_opencv_kpts(dog.detect(cv2.imread(img[0], cv2.IMREAD_GRAYSCALE), None), device=device).to(torch.float)
            laf0 = torch.concat((laf0, dog0), dim=1)

            dog1 = laf_from_opencv_kpts(dog.detect(cv2.imread(img[1], cv2.IMREAD_GRAYSCALE), None), device=device).to(torch.float)
            laf1 = torch.concat((laf1, dog1), dim=1)
        
        # Kornia image load
        timg0 = K.io.load_image(img[0], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)
        timg1 = K.io.load_image(img[1], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)
        
        # patch preprocessing
        if 'AffNet' in patchers:                
            affnet = K.feature.LAFAffNetShapeEstimator(pretrained=True).to(device)
                    
            laf0 = affnet(laf0, timg0)
            laf1 = affnet(laf1, timg1)

        if 'OriNet' in patchers:
            orinet = K.feature.LAFOrienter(angle_detector=K.feature.OriNet(pretrained=True).to(device))

            laf0 = orinet(laf0, timg0)
            laf1 = orinet(laf1, timg1)
            
        # Hardnet
        hardnet = K.feature.LAFDescriptor(patch_descriptor_module=K.feature.HardNet(pretrained=True).to(device))
        
        desc0 = hardnet(timg0, laf0).squeeze(0)
        desc1 = hardnet(timg1, laf1).squeeze(0)
    
        # Keypoints
        kp0 = laf0[:, :, :, 2].to(torch.float).squeeze(0)
        kp1 = laf1[:, :, :, 2].to(torch.float).squeeze(0)
    
        if 'Blob Matching' in matcher:
            # Blob matching (on CPU to avoid OOM)
            m_idx, m_val = dtm.blob_matching(kp0, kp1, desc0, desc1, device='cpu')
            m_idx = m_idx.to(device)
            m_val = m_val.to(device)
            m_mask = torch.ones(m_val.shape[0], device=device, dtype=torch.bool)
        else:
            # Mutual NN matching (with a high threshold)
            th = matcher['Mutual Nearest Neighbor (MNN)']
            m_val, m_idx = K.feature.match_smnn(desc0, desc1, th)
            m_val = m_val.squeeze(1)
            m_mask = torch.ones(m_val.shape[0], device=device, dtype=torch.bool)
    
        # DTM
        match_data = {
            'img': img,
            'kp': [kp0, kp1],
            'm_idx': m_idx,
            'm_val': m_val,
            'm_mask': m_mask,
            }

        # if one just wants to use spatial clues only and not similarity    
        if dtm_only_spatial: match_data['m_val'][:] = 1.0

        # retained matches are signed with 0s in the mask, values > 0 indicate at which iteration the match was discarded   
        dtm_mask = dtm.dtm(match_data, show_in_progress=dtm_show_in_progress, prepare_data=dtm_prepare_data) == 0

        # RANSAC         
        idx = m_idx.to('cpu').detach()
        pt0 = np.ascontiguousarray(kp0.to('cpu').detach())[idx[:, 0]]
        pt1 = np.ascontiguousarray(kp1.to('cpu').detach())[idx[:, 1]]   
        
        F, info = poselib.estimate_fundamental(pt0[dtm_mask], pt1[dtm_mask], poselib_params, {})
        poselib_mask = info['inliers']
        sac_mask = np.copy(dtm_mask)
        sac_mask[dtm_mask] = poselib_mask
        
        # show matches (those discarded by ransac are in red)
        dtm.plot_pair_matches(img, pt0, pt1, dtm_mask, sac_mask)
         
        # the guided filtering can be done zero or more times
        for i in range(ii):
            # re-filter with DTM, guided filtering on previous matches by forcing their similarity values to the lowest value
            match_data['m_val'][sac_mask] = 0
            dtm_mask = dtm.dtm(match_data, show_in_progress=dtm_show_in_progress, prepare_data=dtm_prepare_data) == 0
        
            # RANSAC on re-filtered matches
            idx = m_idx.to('cpu').detach()
            pt0 = np.ascontiguousarray(kp0.to('cpu').detach())[idx[:, 0]]
            pt1 = np.ascontiguousarray(kp1.to('cpu').detach())[idx[:, 1]]   
            
            F, info = poselib.estimate_fundamental(pt0[dtm_mask], pt1[dtm_mask], poselib_params, {})
            poselib_mask = info['inliers']
            sac_mask = np.copy(dtm_mask)
            sac_mask[dtm_mask] = poselib_mask
        
            # show matches
            dtm.plot_pair_matches(img, pt0, pt1, dtm_mask, sac_mask)
            
        # force plots to show if not already happened
        plt.show()
