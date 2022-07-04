# -*- coding: utf-8 -*-
"""
Created on January 2022

@author: Katarina Milicevic, School of Electrical Engineering
         Belgrade, Serbia

Processing of scanned files inside of chosen file directory
"""
import os
import read_all_dicom_series
import scans_preparing


def main(fileDir, workDir):
    
    seriesDir = os.path.join(workDir, "series")

    if not os.path.exists(seriesDir):
        os.makedirs(seriesDir)
    
    flag = 0
    for root, dirs, files in os.walk(fileDir):
        if dirs == []:
            cur = read_all_dicom_series.main(root, seriesDir)
            flag = max(cur, flag)
    
    # flag = read_all_dicom_series.main(fileDir, seriesDir)

    if flag:
        return 1
    flag = scans_preparing.main(seriesDir)
    if flag:
        return 2
   
    return 0
        