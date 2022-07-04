# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:45:12 2021

@author: Katarina Milicevic, School of Electrical Engineering
         Belgrade, Serbia

Segmentation of stone structures in kidneys
"""
import SimpleITK as sitk
import numpy as np
import os


def main(main_dir, phaseIdx, thres=250, thres2=256):
    print("Segm kamena pocela")
    
    if phaseIdx==0:
        img = sitk.ReadImage(main_dir + "/data/delayed_phase_preprocessed.mha")
    elif phaseIdx==1:
        img = sitk.ReadImage(main_dir + "/data/native_phase_preprocessed.mha")
    else:
        print("Greska! Indeks faze nije ni 0 ni 1!")
    img_array = sitk.GetArrayFromImage(img)
    
    try:
        bones_sitk = sitk.ReadImage(main_dir + "/segmentation results/bones.mhd")
    except:
        return 1  
    bones_array = sitk.GetArrayFromImage(bones_sitk)
    only_stone = np.where(bones_array==0, img_array, 0)
    bin_im = (only_stone<thres2)*(only_stone>thres)
    stone = np.zeros(img_array.shape)
    z_top = img_array.shape[0]//6
    z_bottom = 2*img_array.shape[0]//3
    for z in range(z_top, z_bottom+1):
        stone[z,:,:] = bin_im[z,:,:]
    stone_sitk = sitk.GetImageFromArray(stone)
    stone_sitk.SetSpacing(img.GetSpacing())
    stone_sitk = sitk.Cast(stone_sitk, sitk.sitkUInt8)
    cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(stone_sitk, [2, 2, 2])
    stone_sitk = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [2, 2, 2])
    
    # Image saving
    segm_dir = os.path.join(main_dir, "segmentation results")
    if not os.path.exists(segm_dir):
        os.makedirs(segm_dir)
    sitk.WriteImage(stone_sitk, segm_dir + "/stone.mhd")
    
    return 0