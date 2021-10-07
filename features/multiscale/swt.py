# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 13 11:01:34 2021
@reference: Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
Stationary Wavelet Transform (SWT)
==============================================================================
Inputs:
    - f:         image of dimensions N1 x N2
    - mask:      int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                 0 else
    - wavelet:   family of filter for DWT (default='bior3.3')
    - level:     level for DWT decomposition (default=3)
Outputs:
    - features:  mean and std of each cD, cH, cV [9 x 2 = 18]
==============================================================================
"""

import pywt
import numpy as np
from ..utilities import _pad_image_power_2

def swt_features(f, mask, wavelet='bior3.3',levels=3):
        
    # Step 1: Pad image so SWT can work
    f2 = _pad_image_power_2(f)       # pad to the next power of 2 in each dimension
    mask2 = _pad_image_power_2(mask) # pad to the next power of 2 in each dimension
    
    # Step 2: Get SWT Decomposition for 3 levels
    coeffs = pywt.swt2(f2, wavelet, levels)
    
    # Step 3: For each coeff array (10-1 sub-images), get mean and std
    features = np.zeros((3*levels,2),np.double)
    D_mask = mask2.copy()
    i = 0
    for level in range(0,levels):
        coeff = coeffs[level]
        cH, cV, cD = coeff[1]
        for D_f in [cH, cV, cD]:
            D = D_f.flatten()[D_mask.flatten().astype(bool)] # work on elements inside mask
            features[i][0], features[i][1] = abs(D).mean(), abs(D).std()
            i += 1
    
    # Step 4: Create Labels
    labels = []
    for level in range(0, levels):
        for s1 in ['_h','_v','_d']:
            for s2 in ['_mean', '_std']:
                labels.append('SWT_' + str(wavelet) + '_level_' + str(level+1)
                              + str(s1) + str(s2))              
        
    # Step 5: Return
    return features.flatten(), labels