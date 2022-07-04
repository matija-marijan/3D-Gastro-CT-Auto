# -*- coding: utf-8 -*-
"""
Created on January 2022

@author: Katarina Milicevic, School of Electrical Engineering
         Belgrade, Serbia

Making individual phase images with appropriate geometry
"""
import SimpleITK as sitk
import os
from os import walk


def main(imagesDir):
    files = []
    for (dirpath, dirnames, filenames) in walk(imagesDir):
        files.extend(filenames)
        break
    
    nat_image_name = ''
    art_image_name = ''
    del_image_name = ''
    for filename in files:
        if filename.find('NAT')!=-1:
            if nat_image_name!='':
                if os.path.getsize(os.path.join(imagesDir, nat_image_name)) < os.path.getsize(os.path.join(imagesDir, filename)):
                    nat_image_name = filename
            else:
                nat_image_name = filename
        elif filename.find('ART')!=-1:
            if art_image_name!='':
                if os.path.getsize(os.path.join(imagesDir, art_image_name)) < os.path.getsize(os.path.join(imagesDir, filename)):
                    art_image_name = filename
            else:
                art_image_name = filename
        elif filename.find('ODL')!=-1:
            if del_image_name!='':
                if os.path.getsize(os.path.join(imagesDir, del_image_name)) < os.path.getsize(os.path.join(imagesDir, filename)):
                    del_image_name = filename
            else:
                del_image_name = filename
            
    
    # Native phase image preparation
    try:
        nat_image = sitk.ReadImage(os.path.join(imagesDir, nat_image_name))
    except:
        return 1
        
    nat_image_array = sitk.GetArrayFromImage(nat_image)
    
    nat_image_array = nat_image_array[::-1] # flipping

    save_dir = os.path.join(imagesDir, "phases")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    nat_image_new = sitk.GetImageFromArray(nat_image_array)
    nat_image_new.CopyInformation(nat_image)
    sitk.WriteImage(nat_image_new, save_dir+"/native.mha")
                               
             
    # Division of ART/VEN image on separated arterial and vein images and their preparation
    try:
        image = sitk.ReadImage(os.path.join(imagesDir, art_image_name))
        image_array = sitk.GetArrayFromImage(image)
        
        art_image_array = image_array[0:image_array.shape[0]//2,:,:]
        ven_image_array = image_array[image_array.shape[0]//2:,:,:]
        
        art_image = sitk.GetImageFromArray(art_image_array)
        art_image.SetSpacing(nat_image.GetSpacing())
        sitk.WriteImage(art_image, save_dir + "/arterial.mha")
        
        vein_image = sitk.GetImageFromArray(ven_image_array)
        vein_image.SetSpacing(nat_image.GetSpacing())
        sitk.WriteImage(vein_image, save_dir+"/vein.mha")
    except:
        print("Greska sa arterijskom i venskom!")
    
    # Delayed phase preparation
    try:
        del_image = sitk.ReadImage(os.path.join(imagesDir, del_image_name))
    except:
        return 1
            
    del_image_array = sitk.GetArrayFromImage(del_image)
    del_image_array = del_image_array[::-1]
    
    del_image_new = sitk.GetImageFromArray(del_image_array)
    del_image_new.CopyInformation(del_image)
    sitk.WriteImage(del_image_new, save_dir+"/delayed.mha")
    
    return 0