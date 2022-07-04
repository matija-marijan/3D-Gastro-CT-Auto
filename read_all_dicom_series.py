# -*- coding: utf-8 -*-
"""
Created on January 2021

@author: Katarina Milicevic, School of Electrical Engineering
         Belgrade, Serbia

Find and extract individual series from folder with scan files
"""
import SimpleITK as sitk
import os


def main(fileDir, saveDir):
    imageReader = sitk.ImageSeriesReader()
    try:
        seriesIDs = imageReader.GetGDCMSeriesIDs(fileDir)
    except:
        return 1
        
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for i in range(0,len(seriesIDs)):
    
        # dicom_names = reader.GetGDCMSeriesFileNames(sys.argv[1])
        dicom_names = imageReader.GetGDCMSeriesFileNames(fileDir, seriesIDs[i])
        imageReader.SetFileNames(dicom_names)
        
        # Reading phase from one dicom file
        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_names[0])
        reader.ReadImageInformation()
        phase_name_key = "0008|103e"
        phase_name = reader.GetMetaData(phase_name_key)
        print(f"({phase_name_key}) = = \"{phase_name}\"")
        time_key = "0008|0032"
        try:
            time = reader.GetMetaData(time_key)
            print(f"({time_key}) = = \"{time}\"")
        except:
            print("There's no time key!")
         
        image = imageReader.Execute()
        size = image.GetSize()
        print("Image size:", size[0], size[1], size[2])
        
        phase_name = phase_name.replace(" ", "")
        phase_name = phase_name.replace(".", "")
        phase_name = phase_name.replace("/", "-")
        
        outputFileName = saveDir+ "/series" + str(i) + "_" + phase_name + ".mha"
        
        print("Writing image:", outputFileName)

        sitk.WriteImage(image, outputFileName)
        spacing = image.GetSpacing()
        print(f"(spacing) = = \"{spacing}\"")
    
    return 0