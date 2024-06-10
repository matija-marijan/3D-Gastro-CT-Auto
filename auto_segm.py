# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:51:23 2022

@author: Matija Marijan, School of Electrical Engineering
         Belgrade, Serbia
         
Function for automatic segmentation of bones, kidneys (renal collecting system), and kidney stones
"""
import SimpleITK as sitk
import numpy as np
import os
# import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.signal as sci

def main(work_dir):
    
    # plt.close('all')
    
    img_nat = sitk.ReadImage(work_dir + "/data/native_phase_preprocessed.mha")
    img_vein = sitk.ReadImage(work_dir + "/data/vein_phase_preprocessed.mha")
    img_del = sitk.ReadImage(work_dir + "/data/delayed_phase_preprocessed.mha")
    
    arr_nat = sitk.GetArrayFromImage(img_nat)
    arr_vein = sitk.GetArrayFromImage(img_vein)
    arr_del = sitk.GetArrayFromImage(img_del)
    
    #%% Bone segmentation
    
    min_hist = arr_nat.min()
    max_hist = arr_nat.max()
    nat_hist = ndi.histogram(arr_nat, min = min_hist, max = max_hist, bins = max_hist - min_hist + 1)
    nat_hist[0] = 0
    
    # plt.figure()
    # plt.plot(nat_hist)
    # plt.title('Native phase histogram')
    # plt.ylabel('Pixel count')
    # plt.xlabel('Pixel intensity')
    
    try:
        pks, junk = sci.find_peaks(nat_hist, height = 0.5 * max(nat_hist))
    
        start_ind = pks[len(pks) - 1]
        ind = start_ind
    
        while True:
            ind += 1
            if nat_hist[ind] < 0.1 * nat_hist[start_ind]:
                break  
        
        if ind > 0 and ind < 250:
            bone_low_threshold = ind
        else:
            bone_low_threshold = 118
            
    except:
        print("Bone segmentation encountered an error. Segmentation will continue with default parameters.")
        bone_low_threshold = 118
        
    print("bone low threshold = " + str(bone_low_threshold))
    
    # plt.figure()
    # plt.plot(nat_hist)
    # plt.title('Histogram nativne faze')
    # plt.vlines(x = bone_low_threshold, ymin = 0, ymax = max(nat_hist), colors = 'red')
    # plt.vlines(x = 250, ymin = 0, ymax = max(nat_hist), colors = 'red')
    # plt.ylabel('Broj piksela')
    # plt.xlabel('Intenzitet piksela')
    
    bin_im1 = np.array(arr_nat > bone_low_threshold, dtype = 'uint8')    
    bin_im2 = np.array(arr_nat < 250, dtype = 'uint8')
    bone = bin_im1 & bin_im2
    
    bin_im1_del = np.array(arr_del > bone_low_threshold + 10, dtype = 'uint8')
    bin_im2_del = np.array(arr_del < 250, dtype = 'uint8')
    bone_del = bin_im1_del & bin_im2_del
    
    bone_sitk = sitk.GetImageFromArray(bone)
    bone_sitk.SetSpacing(img_vein.GetSpacing())
    bone_sitk = sitk.Cast(bone_sitk, sitk.sitkUInt8)
    
    cleaned_bone = sitk.BinaryOpeningByReconstruction(bone_sitk, [3, 3, 3])
    bone_sitk = sitk.BinaryClosingByReconstruction(cleaned_bone, [5, 5, 5])
    
    dilation_filter = sitk.BinaryDilateImageFilter()
    dilation_filter.SetKernelRadius(3)
    
    bone_dilated = dilation_filter.Execute(bone_sitk)
    bone_dilated.SetSpacing(img_del.GetSpacing())
    
    bone_dilated = sitk.GetArrayFromImage(bone_dilated)
    
    bone = bone_del & bone_dilated
    
    bone_sitk = sitk.GetImageFromArray(bone)
    bone_sitk.SetSpacing(img_vein.GetSpacing())
    bone_sitk = sitk.Cast(bone_sitk, sitk.sitkUInt8)
    
    cleaned_bone = sitk.BinaryOpeningByReconstruction(bone_sitk, [3, 3, 3])
    bone_sitk = sitk.BinaryClosingByReconstruction(cleaned_bone, [5, 5, 5])
    
    #%% Stone segmentation
    
    stone = np.where(bone_dilated == 0, arr_nat, 0)
    stone_threshold = 250
    bin_im3 = (stone > stone_threshold) * (stone < 256)
    stone = np.zeros(arr_nat.shape)

    z_top = arr_del.shape[0]//6
    z_bottom = 2 * arr_del.shape[0]//3
    for z in range(z_top, z_bottom + 1):
        stone[z, :, :] = bin_im3[z, :, :]
    
    stone_sitk = sitk.GetImageFromArray(stone)
    stone_sitk.SetSpacing(img_del.GetSpacing())
    stone_sitk = sitk.Cast(stone_sitk, sitk.sitkUInt8)
    cleaned_stone = sitk.BinaryOpeningByReconstruction(stone_sitk, [2, 2, 2])
    stone_sitk = sitk.BinaryClosingByReconstruction(cleaned_stone, [2, 2, 2])
    
    #%% Liver and Spleen Extraction
    
    vein_no_bone = np.where(bone_dilated == 0, arr_vein, 0)
    
    min_hist = vein_no_bone.min()
    max_hist = vein_no_bone.max()
    vein_hist = ndi.histogram(vein_no_bone, min = min_hist, max = max_hist, bins = max_hist - min_hist + 1)
    vein_hist[0] = 0
    
    # plt.figure()
    # plt.plot(vein_hist)
    # plt.title('Vein phase with no bones histogram')
    # plt.ylabel('Pixel count')
    # plt.xlabel('Pixel intensity')
    
    try:
        pks, junk = sci.find_peaks(vein_hist, height = 0.15 * max(vein_hist))
    
        start_ind = pks[len(pks) - 1]
        up = start_ind
    
        while True:
            up += 1
            if vein_hist[up] < 0.5 * vein_hist[start_ind]:
                break
    
        if up < 250:
            liver_spleen_high_threshold = up
        else:
            liver_spleen_high_threshold = 127
        
    except:
        print("Liver segmentation encountered an error. Segmentation will continue with default parameters.")
        liver_spleen_high_threshold = 127
        
    # plt.figure()
    # plt.plot(vein_hist)
    # plt.title('Histogram venske faze bez kostiju')
    # plt.vlines(x = liver_spleen_high_threshold, ymin = 0, ymax = max(vein_hist), colors = 'red')
    # plt.ylabel('Broj piksela')
    # plt.xlabel('Intenzitet piksela')
        
    print("liver high threshold = " + str(liver_spleen_high_threshold))
    
    #%% Finding Kidney Segmentation Thresholds in Neighborhood of Stones
    
    dilation_filter = sitk.BinaryDilateImageFilter()
    dilation_filter.SetKernelRadius(25)
    
    stone_dilated = dilation_filter.Execute(stone_sitk)
    stone_dilated.SetSpacing(img_del.GetSpacing())
    
    stone_dilated = sitk.GetArrayFromImage(stone_dilated)
    
    kidney_ROI = np.where(stone_dilated == 0, 0, vein_no_bone)
    
    min_hist = kidney_ROI.min()
    max_hist = kidney_ROI.max()
    # max_hist = 255
    kidney_ROI_hist = ndi.histogram(kidney_ROI, min = min_hist, max = max_hist, bins = max_hist - min_hist + 1)
    kidney_ROI_hist[0] = 0
    
    # plt.figure()
    # plt.plot(kidney_ROI_hist)  
    # plt.title('Vein phase kidney ROI histogram')
    # plt.ylabel('Pixel count')
    # plt.xlabel('Pixel intensity')
    
    #%% Kidney Segmentation
    
    try:
        pks, junk = sci.find_peaks(kidney_ROI_hist, height = 0.25 * max(kidney_ROI_hist))
    
        start_ind = pks[len(pks) - 1]
        down = start_ind
        
        while True:
            down -= 1
            if kidney_ROI_hist[down] < 0.4 * kidney_ROI_hist[start_ind]:
                break
    
        if down > 0:
            if down > liver_spleen_high_threshold + 2:
                kidney_low_threshold = down
            else:
                kidney_low_threshold = liver_spleen_high_threshold + 3       
        else:
            kidney_low_threshold = 132
   
    except:
        print("Kidney segmentation encountered an error. Segmentation will continue with default parameters.")
        kidney_low_threshold = 132
   
    print("kidney low threshold = " + str(kidney_low_threshold))
    
    # plt.figure()
    # plt.plot(kidney_ROI_hist)
    # plt.title('Histogram ROI bubrega iz venske faze')
    # plt.vlines(x = kidney_low_threshold, ymin = 0, ymax = max(kidney_ROI_hist), colors = 'red')
    # plt.vlines(x = 250, ymin = 0, ymax = max(kidney_ROI_hist), colors = 'red')
    # plt.ylabel('Broj piksela')
    # plt.xlabel('Intenzitet piksela')
    
    bin_im6 = np.array(arr_del > kidney_low_threshold, dtype = 'uint8')
    bin_im7 = np.array(arr_del < 250, dtype = 'uint8')
    kidney = bin_im6 & bin_im7
    
    kidney = kidney - bone_dilated
    
    kidney_sitk = sitk.GetImageFromArray(kidney)
    kidney_sitk.SetSpacing(img_vein.GetSpacing())
    kidney_sitk = sitk.Cast(kidney_sitk, sitk.sitkUInt8)
    cleaned_kidney = sitk.BinaryOpeningByReconstruction(kidney_sitk, [3, 3, 3])
    kidney_sitk = sitk.BinaryClosingByReconstruction(cleaned_kidney, [3, 3, 3])
    
    #%% Final processing
    
    img_all = bone_sitk * 1 + kidney_sitk * 2 + stone_sitk * 3
    all_array = sitk.GetArrayFromImage(img_all)
    
    # Rotation of axial slices around y-axis
    X_for_mirror = np.transpose(all_array, (0, 2, 1))
    X_mirrored = X_for_mirror[: : -1]
    whole_segm_mirror = np.transpose(X_mirrored, (0, 2, 1))
    
    whole_segm_sitk = sitk.GetImageFromArray(whole_segm_mirror)
    whole_segm_sitk.SetSpacing(img_del.GetSpacing())        
    
    # Segmentation image saving
    save_dir = os.path.join(work_dir, "automatic segmentation results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sitk.WriteImage(whole_segm_sitk, os.path.join(save_dir, "auto_segmentation.mhd"))
    
    # np.save("native", nat_hist)
    # np.save("vein", vein_hist)
    # np.save("kidney", kidney_ROI_hist)
