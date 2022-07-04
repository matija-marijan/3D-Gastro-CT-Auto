# -*- coding: utf-8 -*-
"""
Created on January 2022

@author: Katarina Milicevic, School of Electrical Engineering
         Belgrade, Serbia

Preprocessing of phases
        - Rescaling intensity
        - Median filtering
        - Coregistration
"""
import SimpleITK as sitk
import os


def data_preprocessing_2(img, main_dir, phase_name):

    # Rescaling intensity
    Hmin = -548
    Hmax = 800

    img_255 = sitk.Cast(sitk.IntensityWindowing(img, windowMinimum=Hmin, windowMaximum=Hmax, 
                                             outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)  
    
    # Median filtering
    med_filt = sitk.MedianImageFilter()
    med_filt.SetRadius(1)
    filt_img = med_filt.Execute(img_255) # new_img
    fixed_image_path = os.path.join(main_dir, 'data/native_phase_preprocessed.mha')
    fixed_image =  sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    
    # if (filt_img.GetSize() == fixed_image.GetSize()): # Skipping corregistration if possible
    #     # Image saving
    #     sitk.WriteImage(filt_img, main_dir + "/data/" + phase_name + "_phase_preprocessed.mha")
        
    # else:
        
    # Corregistration
    moving_image = sitk.Cast(filt_img, sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework         
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas= [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                    sitk.Cast(moving_image, sitk.sitkFloat32))
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    casted = sitk.Cast(moving_resampled, sitk.sitkInt16)
    # Image saving
    sitk.WriteImage(casted, os.path.join(main_dir, "data/" + phase_name + "_phase_preprocessed.mha"))


def main(arg, main_dir):

    try:
        img_nat = sitk.ReadImage(arg + "/native.mha")
    except:
            print("Necessary native phase is missing!")
            return 2
    
    # Rescaling intensity
    Hmin = -548
    Hmax = 800 
    img_255 = sitk.Cast(sitk.IntensityWindowing(img_nat, windowMinimum=Hmin, windowMaximum=Hmax, 
                                                 outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)       
 
    # Median filtering
    med_filt = sitk.MedianImageFilter()
    med_filt.SetRadius(1)
    filt_img = med_filt.Execute(img_255)
    
    # Image saving
    data_dir = os.path.join(main_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    sitk.WriteImage(filt_img, os.path.join(data_dir, "native_phase_preprocessed.mha"))
    
    flag = 0;
    
    # Preprocessing of the rest of the phases with corregistration towards native phase if needed
    try:
        img_art = sitk.ReadImage(arg + "/arterial.mha")
        data_preprocessing_2(img_art, main_dir, "arterial")
    except:
        flag = 1;
        print("Arterial phase is missing!")
        
    try:
        img_vein = sitk.ReadImage(arg + "/vein.mha")
        data_preprocessing_2(img_vein, main_dir, "vein")
    except:
        flag = 1;
        print('Vein phase is missing!')
    try:
        img_del = sitk.ReadImage(arg + "/delayed.mha")
        data_preprocessing_2(img_del, main_dir, "delayed")
    except:
        flag = 2;
        print('Delayed phase is missing!')
        
    return flag # 0 for completely sucessful processing
                # 1 for missing unnecessary phases
                # 2 for missing necessary phases