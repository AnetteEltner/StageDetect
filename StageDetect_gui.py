#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Anette Eltner'
__contact__ = 'Anette.Eltner@tu-dresden.de'
__copyright__ = '(c) Anette Eltner 2018'
__license__ = 'MIT'
__date__ = '16 August 2018'
__version__ = '0.1'
__status__ = "initial release"
__url__ = "https://github.com/AnetteEltner/StageDetect"


"""
Name:           StageDetect_gui.py
Compatibility:  Python 2.7
Description:    This program detects water stage using image sequences. It includes
                camera orientation, template matching for GCP detection, master retrieval
                from image sequence, image co-registration, water line detection, and
                transforming 2D points into 3D coordinates. The program has been written in
                cooperation with Melanie Kröhnert and Hannes Sardemann.
URL:            https://github.com/AnetteEltner/StageDetect
Requires:       Tkinter, scipy, scikit-learn, scikit-image, shapely, statsmodels, seaborn,
                cv2 (openCV version 2.4.13)
AUTHOR:         Anette Eltner
ORGANIZATION:   TU Dresden
Contact:        Anette.Eltner@tu-dresden.de
Copyright:      (c) Anette Eltner 2018
Licence:        MIT
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""


import os, csv
import cv2
import numpy as np
import pandas as pd


from Tkinter import *
import tkFileDialog, ScrolledText
from ttk import *

import photogrammetry_classes as georef
import waterline_classes as wl_func
import templateMatching_classes as tmp_match


class WaterlineTool:
    
    def __init__(self, master):
        
        master_frame = Frame(master, name='master_frame')
        master.title('StageDetect: Image-based water level detection')
        note = Notebook(master_frame, name='note')
        master_frame.grid()
        
        #text box for display output
        self.textbox = ScrolledText.ScrolledText(master, height=10, width=20)
        self.textbox.place(x=470, y=30, width=350, height=350)
                                
        
        '''----------------frame waterline in image-------------------'''
        frame = Frame(note)  
        note.add(frame, text="waterline image")
        note.grid(row=0, column=0, ipadx=350, ipady=200)
        
        self.xButton = 370
        self.xText = 250
        self.yAddText = 10
        Style().configure("RB.TButton", foreground='blue', font=('helvetica', 10))
                
        #prepare text box to read parameters
        Label(frame, text="Buffer Size: ").place(x=10, y=self.yAddText)
        self.buff_size_waterlineSearch = IntVar()
        self.buff_size_waterlineSearch_Param = Entry(frame, textvariable=self.buff_size_waterlineSearch)
        self.buff_size_waterlineSearch_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.buff_size_waterlineSearch.set(250)                
        self.OK_ParBuff1 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.buff_size_waterlineSearch))
        self.OK_ParBuff1.place(x=self.xButton, y=self.yAddText, height=20)
        
        self.yAddText = self.yAddText + 30
        Label(frame, text="Threshold histogram: ").place(x=10, y=self.yAddText)  
        self.thresh_hist_waterlineSearch = DoubleVar()
        self.thresh_hist_waterlineSearch_Param = Entry(frame, textvariable=self.thresh_hist_waterlineSearch)
        self.thresh_hist_waterlineSearch_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.thresh_hist_waterlineSearch.set(0.94)                
        self.OK_ParBuff2 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.thresh_hist_waterlineSearch))
        self.OK_ParBuff2.place(x=self.xButton, y=self.yAddText, height=20)
         
        self.yAddText = self.yAddText + 30 
        Label(frame, text="Add value to grey value thresh: ").place(x=10, y=self.yAddText)
        self.add_thresh_grey = IntVar()
        self.add_thresh_grey_Param = Entry(frame, textvariable=self.add_thresh_grey)
        self.add_thresh_grey_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.add_thresh_grey.set(15)                
        self.OK_ParBuff3 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.add_thresh_grey))
        self.OK_ParBuff3.place(x=self.xButton, y=self.yAddText, height=20)
         
        self.yAddText = self.yAddText + 30
        Label(frame, text="Add value to temp texture thresh: ").place(x=10, y=self.yAddText)
        self.add_thresh_tempText = IntVar()
        self.add_thresh_tempText_Param = Entry(frame, textvariable=self.add_thresh_tempText)
        self.add_thresh_tempText_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.add_thresh_tempText.set(55)                
        self.OK_ParBuff4 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.add_thresh_tempText))
        self.OK_ParBuff4.place(x=self.xButton, y=self.yAddText, height=20)
         
        self.yAddText = self.yAddText + 30 
        Label(frame, text="Kernel size bilateral filter: ").place(x=10, y=self.yAddText)
        self.bilat_kernelStr = StringVar()
        self.bilat_kernel_Param = Entry(frame, textvariable=self.bilat_kernelStr)
        self.bilat_kernel_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.bilat_kernelStr.set("7:3")                
        self.OK_ParBuff5 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.bilat_kernelStr))
        self.OK_ParBuff5.place(x=self.xButton, y=self.yAddText, height=20)
        
        self.yAddText = self.yAddText + 30 
        Label(frame, text="Kernel size canny filter: ").place(x=10, y=self.yAddText)
        self.canny_kernel = IntVar()
        self.canny_kernel_Param = Entry(frame, textvariable=self.canny_kernel)
        self.canny_kernel_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.canny_kernel.set(3)                
        self.OK_ParBuff6 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.canny_kernel))
        self.OK_ParBuff6.place(x=self.xButton, y=self.yAddText, height=20)
                
        self.yAddText = self.yAddText + 30
        Label(frame, text="Clip size NaN values: ").place(x=10, y=self.yAddText)
        self.nan_clip_size = IntVar()
        self.nan_clip_size_Param = Entry(frame, textvariable=self.nan_clip_size)
        self.nan_clip_size_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.nan_clip_size.set(10)                
        self.OK_ParBuff7 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.nan_clip_size))
        self.OK_ParBuff7.place(x=self.xButton, y=self.yAddText, height=20)     
        
        self.yAddText = self.yAddText + 30
        Label(frame, text="Start at image: ").place(x=10, y=self.yAddText)
        self.video_value = StringVar()
        self.video_value_Param = Entry(frame, textvariable=self.video_value)
        self.video_value_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.video_value.set(" ")                
        self.OK_ParBuff8 = Button(frame, text="Get", command= lambda: self.getValuesFromTxtBox(self.video_value))
        self.OK_ParBuff8.place(x=self.xButton, y=self.yAddText, height=20) 


        #check waterside
        self.yAddText = self.yAddText + 30
        Label(frame, text="Waterside: ").place(x=10, y=self.yAddText)
        self.watersideInt = IntVar()
        self.watersideInt.set(0)
        self.watersideBut = Radiobutton(frame, text = "Left", variable=self.watersideInt, value=0)
        self.watersideBut.place(x=80, y=self.yAddText, height=20)
        self.watersideBut = Radiobutton(frame, text = "Right", variable=self.watersideInt, value=1)
        self.watersideBut.place(x=130, y=self.yAddText, height=20)
         
        #check if Canny should be used, else region growing depending on color is used
        self.yAddText = self.yAddText + 20
        self.use_canny = BooleanVar()
        self.use_canny.set(True)
        self.checkCannyBut = Checkbutton(frame, text = "Use Canny filter", variable=self.use_canny)
        self.checkCannyBut.place(x=10, y=self.yAddText)          
          
        #check if results should be illustrated
        self.plot_results = BooleanVar()
        self.plot_results.set(False)
        self.plot_resultsBut = Checkbutton(frame, text = "Plot results", variable=self.plot_results)
        self.plot_resultsBut.place(x=180, y=self.yAddText)
          
        #check if co-registration should be performed
        self.yAddText = self.yAddText + 20
        self.perform_coregist = BooleanVar()
        self.perform_coregist.set(True)
        self.perform_coregistBut = Checkbutton(frame, text = "Perform co-registration", variable=self.perform_coregist)
        self.perform_coregistBut.place(x=10, y=self.yAddText)
                    
        #check if start from specific video
        self.do_continue = BooleanVar()
        self.do_continue.set(False)
        self.do_continueBut = Checkbutton(frame, text = "Start from specific image", variable=self.do_continue)
        self.do_continueBut.place(x=180, y=self.yAddText)          
          
        #check whether approximation only one file or list of files
        self.yAddText = self.yAddText + 20
        self.waterline_approx_steady = BooleanVar()
        self.waterline_approx_steady.set(False)
        self.checkWaterlineApproxNbr = Checkbutton(frame, text = "Waterline approx: single file", variable=self.waterline_approx_steady)
        self.checkWaterlineApproxNbr.place(x=10, y=self.yAddText)
        
        #check whether processing only for one waterline
        self.yAddText = self.yAddText + 20
        self.waterline_single = BooleanVar()
        self.waterline_single.set(False)
        self.checkWaterlineSingle = Checkbutton(frame, text = "Waterline detection: single case", variable=self.waterline_single)
        self.checkWaterlineSingle.place(x=10, y=self.yAddText)
          
          
        #prepare starting waterline detection button
        self.waterlineDetection = Button(frame, text="Detect waterline", style="RB.TButton", command=self.waterlineDetection)
        self.waterlineDetection.place(x=10, y=self.yAddText+30)

                        
        
        '''----------------get GCP coordinates in image-------------------'''
        frame2 = Frame(note)  
        note.add(frame2, text="get GCP image coordinates")
        note.grid(row=0, column=0, ipadx=350, ipady=200)
        
        self.yAddText = 10     
        self.xButton = 250
        self.xText = 180

        #prepare image list with masters        
        Label(frame2, text="Variable folder search: ").place(x=10, y=self.yAddText)
        self.varDirSearch = StringVar()
        self.varDirSearch_Param = Entry(frame2, textvariable=self.varDirSearch)
        self.varDirSearch_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.varDirSearch.set('2017')                
        self.OK_ParBuff18 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.varDirSearch))
        self.OK_ParBuff18.place(x=self.xButton, y=self.yAddText, height=20, width=50)
        
        self.yAddText = self.yAddText + 30
        Label(frame2, text="Variable sub folder search: ").place(x=10, y=self.yAddText)
        self.varSubDirSearch = StringVar()
        self.varSubDirSearch_Param = Entry(frame2, textvariable=self.varSubDirSearch)
        self.varSubDirSearch_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.varSubDirSearch.set('_0.jpg')                
        self.OK_ParBuff17 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.varSubDirSearch))
        self.OK_ParBuff17.place(x=self.xButton, y=self.yAddText, height=20, width=50)
             
        self.yAddText = self.yAddText + 30
        self.imgList = Button(frame2, text="Image master list", style="RB.TButton", command=self.getImgList)
        self.imgList.place(x=10, y=self.yAddText)
                
                
        #set parameters for template matching
        self.yAddText = self.yAddText + 70
        Label(frame2, text="Template size x: ").place(x=10, y=self.yAddText)
        self.template_size_x = IntVar()
        self.template_size_x_Param = Entry(frame2, textvariable=self.template_size_x)
        self.template_size_x_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.template_size_x.set(300)                
        self.OK_ParBuff11 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.template_size_x))
        self.OK_ParBuff11.place(x=self.xButton, y=self.yAddText, height=20, width=50)
        
        self.yAddText = self.yAddText + 30
        Label(frame2, text="Template size y: ").place(x=10, y=self.yAddText)
        self.template_size_y = IntVar()
        self.template_size_y_Param = Entry(frame2, textvariable=self.template_size_y)
        self.template_size_y_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.template_size_y.set(300)                
        self.OK_ParBuff12 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.template_size_y))
        self.OK_ParBuff12.place(x=self.xButton, y=self.yAddText, height=20, width=50)       
        
        self.yAddText = self.yAddText + 30
        Label(frame2, text="Search area x: ").place(x=10, y=self.yAddText)
        self.search_area_x = IntVar()
        self.search_area_x_Param = Entry(frame2, textvariable=self.search_area_x)
        self.search_area_x_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.search_area_x.set(500)                
        self.OK_ParBuff13 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.search_area_x))
        self.OK_ParBuff13.place(x=self.xButton, y=self.yAddText, height=20, width=50)
        
        self.yAddText = self.yAddText + 30
        Label(frame2, text="Search area y: ").place(x=10, y=self.yAddText)
        self.search_area_y = IntVar()
        self.search_area_y_Param = Entry(frame2, textvariable=self.search_area_y)
        self.search_area_y_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.search_area_y.set(500)                
        self.OK_ParBuff14 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.search_area_y))
        self.OK_ParBuff14.place(x=self.xButton, y=self.yAddText, height=20, width=50)
        
        self.yAddText = self.yAddText + 30
        Label(frame2, text="Error threshold (%): ").place(x=10, y=self.yAddText)
        self.error_accpt = DoubleVar()
        self.error_accpt_Param = Entry(frame2, textvariable=self.error_accpt)
        self.error_accpt_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.error_accpt.set(0.01)                
        self.OK_ParBuff15 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.error_accpt))
        self.OK_ParBuff15.place(x=self.xButton, y=self.yAddText, height=20, width=50)
        
        self.yAddText = self.yAddText + 30
        Label(frame2, text="Maximum points skip able: ").place(x=10, y=self.yAddText)
        self.max_ptsToSkip = IntVar()
        self.max_ptsToSkip_Param = Entry(frame2, textvariable=self.max_ptsToSkip)
        self.max_ptsToSkip_Param.place(x=self.xText, y=self.yAddText, width=60, height=20)
        self.max_ptsToSkip.set(5)                
        self.OK_ParBuff16 = Button(frame2, text="Get", command= lambda: self.getValuesFromTxtBox(self.max_ptsToSkip))
        self.OK_ParBuff16.place(x=self.xButton, y=self.yAddText, height=20, width=50)
        
        #check if results should be illustrated
        self.yAddText = self.yAddText + 30
        self.plot_results2 = BooleanVar()
        self.plot_results2.set(False)
        self.plot_resultsBut2 = Checkbutton(frame2, text = "Plot results", variable=self.plot_results2)
        self.plot_resultsBut2.place(x=10, y=self.yAddText)
                    
        #check if save images with matched points
        self.save_img = BooleanVar()
        self.save_img.set(False)
        self.save_imgBut = Checkbutton(frame2, text = "Save image with matches", variable=self.save_img)
        self.save_imgBut.place(x=180, y=self.yAddText)
        
        self.templateMatch = Button(frame2, text="Template matching", style="RB.TButton", command=self.performTemplateMatching)
        self.templateMatch.place(x=10, y=self.yAddText+30)
        
        
        '''----------------get location approximated waterlines (account for camera movement)-------------------'''
        frame3 = Frame(note)  
        note.add(frame3, text="2D to 3D (and again 2D)")
        note.grid(row=0, column=0, ipadx=350, ipady=200)
        
        self.yAddText = 10     
        self.xButton = 390
        self.xText = 290
        
        Label(frame3, text="Exterior orientation (estimates): ").place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        self.exterior_approx_raw = StringVar()
        self.exterior_approx_raw_Param = Entry(frame3, textvariable=self.exterior_approx_raw)
        self.exterior_approx_raw_Param.place(x=10, y=self.yAddText, width=300, height=20)
        self.exterior_approx_raw.set('281, 4288, 2255, -0.053, -0.857, -1.908')                
        self.OK_ParBuff31 = Button(frame3, text="Get", command= lambda: self.getValuesFromTxtBox(self.exterior_approx_raw))
        self.OK_ParBuff31.place(x=320, y=self.yAddText, height=20, width=50)
        
        self.yAddText = self.yAddText + 30
        Label(frame3, text="Unit GCP (mm): ").place(x=10, y=self.yAddText)  
        self.unit_gcp = IntVar()
        self.unit_gcp_Param = Entry(frame3, textvariable=self.unit_gcp)
        self.unit_gcp_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.unit_gcp.set(1000)                
        self.OK_ParBuff32 = Button(frame3, text="Get", command= lambda: self.getValuesFromTxtBox(self.unit_gcp))
        self.OK_ParBuff32.place(x=self.xButton, y=self.yAddText, height=20, width=50)
         
        self.yAddText = self.yAddText + 30 
        Label(frame3, text="Minimum sigma 0 (resection): ").place(x=10, y=self.yAddText)
        self.min_s0 = DoubleVar()
        self.min_s0_Param = Entry(frame3, textvariable=self.min_s0)
        self.min_s0_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.min_s0.set(0.1)                
        self.OK_ParBuff33 = Button(frame3, text="Get", command= lambda: self.getValuesFromTxtBox(self.min_s0))
        self.OK_ParBuff33.place(x=self.xButton, y=self.yAddText, height=20, width=50)
         
        self.yAddText = self.yAddText + 30
        Label(frame3, text="Skip value (prior starting): ").place(x=10, y=self.yAddText)
        self.skip_val = IntVar()
        self.skip_val_Param = Entry(frame3, textvariable=self.skip_val)
        self.skip_val_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.skip_val.set(0)                
        self.OK_ParBuff34 = Button(frame3, text="Get", command= lambda: self.getValuesFromTxtBox(self.skip_val))
        self.OK_ParBuff34.place(x=self.xButton, y=self.yAddText, height=20, width=50)
         
        self.yAddText = self.yAddText + 30 
        Label(frame3, text="Maximum distance NN search (pix): ").place(x=10, y=self.yAddText)
        self.max_NN_dist = IntVar()
        self.max_NN_dist_Param = Entry(frame3, textvariable=self.max_NN_dist)
        self.max_NN_dist_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.max_NN_dist.set(1)                
        self.OK_ParBuff35 = Button(frame3, text="Get", command= lambda: self.getValuesFromTxtBox(self.max_NN_dist))
        self.OK_ParBuff35.place(x=self.xButton, y=self.yAddText, height=20, width=50)
        
        self.yAddText = self.yAddText + 30 
        Label(frame3, text="Maximum orientation difference to estimates: ").place(x=10, y=self.yAddText)
        self.max_orient_diff = DoubleVar()
        self.max_orient_diff_Param = Entry(frame3, textvariable=self.max_orient_diff)
        self.max_orient_diff_Param.place(x=self.xText, y=self.yAddText, width=100, height=20)
        self.max_orient_diff.set(0.1)                
        self.OK_ParBuff36 = Button(frame3, text="Get", command= lambda: self.getValuesFromTxtBox(self.max_orient_diff))
        self.OK_ParBuff36.place(x=self.xButton, y=self.yAddText, height=20, width=50)

                
        #check if results should be illustrated
        self.yAddText = self.yAddText + 30
        self.plot_results3 = BooleanVar()
        self.plot_results3.set(False)
        self.plot_resultsBut3 = Checkbutton(frame3, text = "Plot results", variable=self.plot_results3)
        self.plot_resultsBut3.place(x=10, y=self.yAddText)
                    
        #check if RANSAC used for approximation values
        self.use_ransac_for_approx = BooleanVar()
        self.use_ransac_for_approx.set(False)
        self.use_ransac_for_approxBut = Checkbutton(frame3, text = "RANSAC for estimates", variable=self.use_ransac_for_approx)
        self.use_ransac_for_approxBut.place(x=100, y=self.yAddText)
               
        
        self.yAddText = self.yAddText + 30
        Label(frame3, text="Waterline approximation or water level retrieval: ").place(x=10, y=self.yAddText)
        self.waterlineApprox3D = IntVar()
        self.waterlineApprox3D.set(0)
        self.yAddText = self.yAddText + 20
        self.waterlineApprox3DBut = Radiobutton(frame3, text = "Water line Approx", variable=self.waterlineApprox3D, value=1)
        self.waterlineApprox3DBut.place(x=10, y=self.yAddText, height=20)
        self.waterlineApprox3DBut = Radiobutton(frame3, text = "Water level Retrieval", variable=self.waterlineApprox3D, value=0)
        self.waterlineApprox3DBut.place(x=150, y=self.yAddText, height=20)        
        
        
        self.templateMatch = Button(frame3, text="Get 3D from 2D", style="RB.TButton", command=self.get3Dfrom2D)
        self.templateMatch.place(x=10, y=self.yAddText+30)
             
        
    def get3Dfrom2D(self):
        
        '''----read input----'''
        #read parameters from GUI
        try:
            use_ransac_for_approx = self.use_ransac_for_approx.get()
            plot_results = self.plot_results3.get()
            
            exterior_approx_rawStr = self.exterior_approx_raw.get()
            exterior_approx_raw = exterior_approx_rawStr.split(',')
            exterior_approx_raw = np.asarray([float(x) for x in exterior_approx_raw]).reshape(6,1)
            unit_gcp = self.unit_gcp.get()
            min_s0 = self.min_s0.get()
            skip_val = self.skip_val.get()
            max_NN_dist = self.max_NN_dist.get()
            max_orient_diff = self.max_orient_diff.get()
        
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)            
            self.printTxt('retrieving parameters failed\n')        
        
        
        #read parameters from directories
        failing = True
        while failing:
            try:
                dir_output = tkFileDialog.askdirectory(title='Output directory')
                dir_output = dir_output + '/'
                dir_imgCooGCP = tkFileDialog.askdirectory(title='Directory of GCP image coordinates')
                dir_imgCooGCP = dir_imgCooGCP + '/'
                
                if self.waterlineApprox3D.get() == 1:
                    waterline_file = tkFileDialog.askopenfilename(title='File with initial waterline', 
                                                                  filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
                else:
                    directory_waterline = tkFileDialog.askdirectory(title='Directory of waterlines') + '/'
                    
                GPC_coo_file = tkFileDialog.askopenfilename(title='File with GCP coordinates (3D)', 
                                                            filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
                ior_file = tkFileDialog.askopenfilename(title='Read interior orientation file', 
                                                        filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
                model_3Dpts = tkFileDialog.askopenfilename(title='Read 3D point cloud (XYZ)', 
                                                           filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())

                failing = False
            
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                self.printTxt('failed reading directories, please try again\n')
                self.dummy()


        '''----start processing----'''
        try:
            '''prepare output'''
            #write files ransac result
            if use_ransac_for_approx:
                ransac_result_file = open(dir_output + 'ransac_result.txt', 'wb')
                writer = csv.writer(ransac_result_file, delimiter=' ')
                writer.writerow(['datetime', 'X', 'Y', 'Z', 'nbr_inliers', 'outliers'])

            #write files orientation result
            calib_results_output = ['image', 's0', 'X','stdX', 'Y', 'stdY', 'Z', 'stdZ',
                                    'omega', 'stdomega', 'phi', 'stdphi', 'kappa', 'stdkappa']
            calib_results_output = pd.DataFrame(calib_results_output).T
            calib_results_output.to_csv(dir_output + 'AccuracyCamOrient.txt', sep='\t', index=False, header=False)
            
            #write files water level result
            ouput_waterlevel_file = open(dir_output + 'waterlevel.txt', 'wb')
            writer_waterlevel =  csv.writer(ouput_waterlevel_file, delimiter=' ')
            
            #write output 3D measurement
            if self.waterlineApprox3D.get() == 1:                  
                file_3D = open(dir_output + 'Waterline_Approx_3D.txt', 'wb')
            else:
                file_3D = open(dir_output + 'Waterline_3D.txt', 'wb')            
            
            
            '''read measurements'''
            if self.waterlineApprox3D.get() == 1:
                #read waterline file
                waterline_table = pd.read_table(waterline_file, header=None, delimiter=',')
                waterline = np.asarray(waterline_table)            
            
            #read interior orientation from file (aicon)
            interior_orient = georef.camera_interior()
            interior_orient.read_aicon_ior(ior_file)
            
            #convert camera parameters to pixel value for opencv
            pixel_size = interior_orient.sensor_size_x / interior_orient.resolution_x
            ck = -1 * interior_orient.ck / pixel_size
            xh = interior_orient.resolution_x / 2
            yh = interior_orient.resolution_y / 2
            cam_file_forOpenCV = [ck, xh, yh, 0, 0, 0, 0, 0]            
            
            #read point cloud
            pt_cloud_table = pd.read_csv(model_3Dpts, header=None, index_col=False, delimiter=',')
            ptCloud = np.asarray(pt_cloud_table,dtype=np.float)
            del pt_cloud_table
                        
            #read object coordinates of GCP (including point ID)
            gcp_table = pd.read_csv(GPC_coo_file, header=None, index_col=False, delimiter='\t')
            gcp_table = np.asarray(gcp_table,dtype=np.float)
            gcp_table[:,1:4] = gcp_table[:,1:4] * unit_gcp

            
            '''prepare water line data'''                   
            #get water line dates
            waterline_dates = []
            for imgGCPCoo_file in os.listdir(dir_imgCooGCP):
                if 'imgPtsGCP_' in imgGCPCoo_file:
                    waterline_date = imgGCPCoo_file[10:-4]
                    waterline_dates.append(waterline_date)
            waterline_dates = sorted(waterline_dates)
                
            #search for waterline files
            if self.waterlineApprox3D.get() == 0:
                #undistort image measurements of waterline            
                waterline_files = []
                if os.path.isdir(directory_waterline):
                    for file_line in os.listdir(directory_waterline):
                        if 'wasserlinie' in file_line:
                            waterline_files.append(file_line)
                waterline_files =  sorted(waterline_files, key=lambda waterline_order: waterline_order)
 
 
            '''start 3D retrieval'''
            position_ref = 0
            nbr_img = 0   
            XYZ_estim = False
            waterline_3D_approx_given = False
            skip_val = skip_val - 1        
            waterline_found = False
            
            img_measures = georef.image_measures()
            waterline_pt_cl = georef.PtImg()
            pts_to_undist_cl = georef.PtImg()
            cl_2D_3D = georef.TwoD_to_ThreeD()
            conversionMat = georef.conversions()
            pt3D = georef.Pt3D()
            waterline_xy = georef.PtImg()
            
            while nbr_img < len(waterline_dates): 
                
                waterline_date = str(waterline_dates[nbr_img])    #str(waterline[0])    in case of longer date name
                
                '''read data for each iteration'''
                #read water line if list of water lines
                if self.waterlineApprox3D.get() == 0:
                    for waterline_file in waterline_files:
                        if waterline_date == waterline_file[12:-4]:
                            waterline_table = pd.read_table(directory_waterline + waterline_file, header=None, delimiter=',')
                            waterline = np.asarray(waterline_table)
                            waterline_found = True
                            break
                    if not waterline_found:
                        nbr_img = nbr_img + 1
                        skip_val = skip_val + 1
                        print('no waterline file given')          
                        continue    
                
                #read pixel coordinates from corresponding files of template matching results    
                pts_table = pd.read_csv(dir_imgCooGCP + 'imgPtsGCP_' + str(waterline_dates[nbr_img] + '.txt'),
                                        header=None, index_col=False, delimiter='\t') #'_ellipse' +
                pts_table = np.asarray(pts_table, dtype=np.float)
                pts_ids = pts_table[:,0]
                pts_ids = pts_ids.reshape(pts_ids.shape[0],1)
                pts_to_undist = pts_table[:,1:3]
             
                self.printTxt('process ' + waterline_date + '\n')
                self.dummy()
                 
                 
                '''undistort image measurements'''
                #undistort waterline measurement
                waterline_pt = waterline[:,:]
                waterline_pt_cl.read_imgPts(waterline_pt)
                
                waterline_pts_undist = img_measures.undistort_img_coos(waterline_pt_cl, interior_orient)
                waterline_pts_undist_px = img_measures.metric_to_pixel(waterline_pts_undist, interior_orient)
                 
                #undistort image measurements of GCP measurements
                pts_to_undist_cl.read_imgPts(pts_to_undist)
                img_pts_undist_metric = img_measures.undistort_img_coos(pts_to_undist_cl, interior_orient, False)               
             
             
                '''re-organise coordinates to numpy matrix with assigned pt ids'''
                img_pts_undist_metric_id = np.hstack((pts_ids, img_pts_undist_metric.x.reshape(img_pts_undist_metric.x.shape[0],1)))
                img_pts_undist_metric_id = np.hstack((img_pts_undist_metric_id, img_pts_undist_metric.y.reshape(img_pts_undist_metric.y.shape[0],1)))
                ImgGCPCoo, _ = cl_2D_3D.assign_ImgToObj_Measurement(gcp_table, img_pts_undist_metric_id)     
                  
              
                '''get exterior camera geometry''' 
                if use_ransac_for_approx: 
                    '''using RANSAC in OpenCV'''
                    #convert image measurements into pixels for opencv
                    img_pts_undist = img_measures.metric_to_pixel(img_pts_undist_metric, interior_orient)
                    img_pts_undist = np.hstack((pts_ids, img_pts_undist))
                    ImgGCPCoo_pix, _ = cl_2D_3D.assign_ImgToObj_Measurement(gcp_table, img_pts_undist)
                   
                    #get camera position with OpenCV
                    rot_mat, position, inliers = cl_2D_3D.image_orientation_RANSAC(ImgGCPCoo_pix, cam_file_forOpenCV)   #True, img_to_read
                     
                    #convert rot_mat into angles
                    conversionMat = georef.conversions()
                    exterior_approx = conversionMat.rotMat_to_angle(rot_mat, position)
                    self.printTxt(exterior_approx)
                    self.dummy()
                    
                    if inliers == None:
                        inliers = [-999]
                        writer.writerow([waterline_date, position[0,0], position[1,0], position[2,0], '-'])
                    else:
                        writer.writerow([waterline_date, position[0,0], position[1,0], position[2,0], len(inliers)])
                    ransac_result_file.flush()
                
                 
                '''using resection with adjustment'''
                cam_file_forResection = [interior_orient.ck, interior_orient.xh, interior_orient.yh,    #note that ck is negative (used unchanged from aicon)
                                         0, 0, 0, 0, 0, 0, 0, 0]
                if not use_ransac_for_approx:
                    exterior_approx = np.zeros((exterior_approx_raw.shape[0],1))
                    exterior_approx[:] = exterior_approx_raw[:]
                
                resection = georef.resection()
                    
                calib_results, s0 = resection.resection(cam_file_forResection, exterior_approx, ImgGCPCoo, 0.00001, plot_results)
                if not calib_results[0,0] == -9999:
                    #print(calib_results)
                        
                    position = calib_results[0:3,0] / unit_gcp
                        
                    # rotation = calib_results[3:6,0]        
                    #convert angles into rotation matrix
                    rot_mat = conversionMat.rot_Matrix(calib_results[3,0], calib_results[4,0], calib_results[5,0], 'radians').T
                    multipl_array = np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
                    rot_mat = rot_mat * multipl_array
                    
                    accuracyCamOrient_output = [waterline_date] + [s0[0,0]] + calib_results.flatten().tolist()
                    accuracyCamOrient_output = pd.DataFrame(accuracyCamOrient_output).T
                    accuracyCamOrient_output.to_csv(dir_output + 'AccuracyCamOrient.txt', mode='a', sep='\t', index=False, header=False)

                else:
                    self.printTxt('referencing skipped\n')
                    nbr_img = nbr_img + 1
                    skip_val = skip_val + 1
                    calib_results_output = [waterline_date, -9999]
                    calib_results_output = pd.DataFrame(calib_results_output).T
                    calib_results_output.to_csv(dir_output + 'AccuracyCamOrient.txt', mode='a', sep='\t', index=False) 
                    continue
            
                #process only waterlines where referencing at least within 90% of good registration
                if nbr_img == skip_val:
                    position_ref_neg = position - max_orient_diff * position   
                    position_ref_pos = position + max_orient_diff * position   
                    position_ref = 1
                    print('orient range: ' + str(position_ref_neg) + str(position_ref_pos))
                if position_ref == 1 and nbr_img > skip_val:
                    if (position_ref_neg[0] > position[0] or position_ref_pos[0] < position[0] or
                        position_ref_neg[1] > position[1] or position_ref_pos[1] < position[1] or
                        position_ref_neg[2] > position[2] or position_ref_pos[2] < position[2]):        
                        print('orientation too large deviations')
                        nbr_img = nbr_img + 1 
                        continue
                 
                eor_mat = np.hstack((rot_mat.T, position.reshape(position.shape[0],1))) #if rotation matrix received from opencv transpose rot_mat
                eor_mat = np.vstack((eor_mat, [0,0,0,1]))
                 
                if position[0] < 0 or position[1] < 0 or position[2] < 0:   #projection center needs to be positive
                    print('failed image referencing')
                    nbr_img = nbr_img + 1 
                    continue
                 
                print('image referenced: ' + str(position))
                
                 
                '''project into image space'''
                if self.waterlineApprox3D.get() == 0:
                    waterline_3D_approx_given = False
                
                if not waterline_3D_approx_given:  
                    if nbr_img > skip_val and s0 < min_s0:  #minimum quality of resection needed        
                        #project points into depth image
                        if ptCloud.shape[1] > 3:
                            pt3D.rgb = True
                        
                        pt3D.read_imgPts_3D(ptCloud)
                        
                        try: 
                            xyd_rgb_map = cl_2D_3D.project_pts_into_img(eor_mat, interior_orient, pt3D, plot_results,False)
                        except Exception as e:
                            print(e)
                            print('registration image erroneous')
                            nbr_img = nbr_img + 1 
                            continue
                             
                        print('point cloud projected into img')  
                         
                        #find nearest depth value to waterline in depth image
                        waterline_xyz = cl_2D_3D.NN_pts(xyd_rgb_map, waterline_pts_undist_px, max_NN_dist, False)
                        if waterline_xyz == None:
                            print('no NN for waterlevel')
                            nbr_img = nbr_img + 1 
                            continue
                        print('nearest neighbours found')
                                                
                        
                        '''project into object space again'''  
                        #transform image measurement into object space
                        try:
                            waterline_xy.x = waterline_xyz.X
                            waterline_xy.y = waterline_xyz.Y
                            imgPts_mm = img_measures.pixel_to_metric(waterline_xy, interior_orient)
                        except Exception as e:
                            print(e)
                            nbr_img = nbr_img + 1 
                            continue    
                        
                        waterline_xyz.X = imgPts_mm.x
                        waterline_xyz.Y = imgPts_mm.y
            
                        xyd_map_mm = cl_2D_3D.imgDepthPts_to_objSpace(waterline_xyz, eor_mat, interior_orient)
                          
                        #write output
                        xyd_map_mm_write = np.hstack((xyd_map_mm.X.reshape(xyd_map_mm.X.shape[0],1), xyd_map_mm.Y.reshape(xyd_map_mm.Y.shape[0],1)))
                        xyd_map_mm_write = np.hstack((xyd_map_mm_write, xyd_map_mm.Z.reshape(xyd_map_mm.Z.shape[0],1))) 
                        writer = csv.writer(file_3D, delimiter=' ')
                        writer.writerows(xyd_map_mm_write)
                        file_3D.flush()
                        
                        XYZ_estim = True
                        waterline_3D_approx_given = True
                    
                    if nbr_img == skip_val and s0 >= min_s0:
                        skip_val = skip_val + 1
                        nbr_img = nbr_img + 1
                        continue
                    
                elif skip_val > nbr_img:
                    nbr_img = nbr_img + 1
                    continue
                
                
                '''re-project found approx-waterline 3D coordinates into image space of each frame to account for camera movements and get adopted approx position'''
                if self.waterlineApprox3D.get() == 1:
                    if nbr_img > skip_val and s0 < min_s0 and XYZ_estim == True:  #minimum quality of resection needed 
                        #project points into image
                        try:
                            xy_waterline_approx = cl_2D_3D.project_pts_into_img(eor_mat, interior_orient, xyd_map_mm, plot_results)
                            # xy_waterline_approx_xy = np.hstack((xy_waterline_approx.X, xy_waterline_approx.Y))
                        except:
                            print('registration image erroneous. waterline approx 3D not projectable into image space.')
                            nbr_img = nbr_img + 1 
                            continue
                         
                        print('waterline approx 3D projected into img')
                        
                        #write output
                        log_file_temp = open(dir_output + waterline_date + 'waterline_approx.txt', 'wb')
                        writer2 = csv.writer(log_file_temp, delimiter=',')
                        xy_waterline_approx_write = np.hstack((xy_waterline_approx.X.reshape(xy_waterline_approx.X.shape[0],1), 
                                                               xy_waterline_approx.Y.reshape(xy_waterline_approx.Y.shape[0],1)))                        
                        writer2.writerows(xy_waterline_approx_write)
                        log_file_temp.flush()
                        log_file_temp.close()
                    
                    else:
                        nbr_img = nbr_img + 1
                        continue            
                
                else:
                    '''calculate median of all height values to get one waterlevel measurement'''
                    waterlevel_mean = np.mean(xyd_map_mm.Z)
                    waterlevel_median = np.median(xyd_map_mm.Z)
                    waterlevel_std = np.std(xyd_map_mm.Z)
                    waterlevel_min = np.min(xyd_map_mm.Z)
                    waterlevel_max = np.max(xyd_map_mm.Z)
                    writer_waterlevel.writerow([waterline_date, waterlevel_mean, waterlevel_median, waterlevel_std, waterlevel_min, waterlevel_max, s0[0,0]])
                    ouput_waterlevel_file.flush()
                
                nbr_img = nbr_img + 1   
        
            self.printTxt('getting waterline approxmation finisched successfully\n') 
        
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            self.printTxt('getting waterline approxmation failed\n')  
            

    def printTxt(self, txt):
        self.textbox.insert(END, txt)
        return

    def dummy(self):
        self.textbox.insert(END, ' ')
        return

    def getImgList(self):
        
        failing = True
        while failing:
            try:
                directory_folders = tkFileDialog.askdirectory(title='Directory to search in') 
#                 if type(directory_folders) == unicode:
#                     directory_folders = directory_folders.encode('ascii','ignore')
                directory_results = tkFileDialog.askdirectory(title='Output Directory')
                failing = False
            except Exception as e:
                print(e)
                self.printTxt('failed reading directories, please try again\n')
        
        #search through directory for sub-directories
        try:
            self.printTxt('start image retrieval\n')
            
            dir_folders = []
            var_dirsearch = self.varDirSearch.get()
            var_subdirsearch = self.varSubDirSearch.get()
            if os.path.isdir(directory_folders):
                for dirpath, dirsubpaths, dirfiles in os.walk(directory_folders):                    
                    for folder in dirsubpaths:                        
                        if var_dirsearch in folder:   #'2017', 'coreg'
            #                 dir_folders.append(folder)
                            
                            for files in os.listdir(dirpath + '/' + folder + '/'):    #
                                if var_subdirsearch in files:      #'0_coreg.jpg'
                                    if type(files) == unicode:
                                        files = files.encode('ascii','ignore')
                                    dir_folders.append(files[:-4]) #files[:-13], files[:-12]
                                    
                                    # os.system('cp -r -f ' + dirpath + '/coreg/' + files + ' ' + directory_results)
                                    os.system('cp -r -f ' + dirpath + '/' + folder + '/' + files + ' ' + directory_results)
            
            folder_list =  sorted(dir_folders, key=lambda folder_order: folder_order)
            
            img_file_write = open(directory_results + 'img_list.txt', 'wb')
            writer_img = csv.writer(img_file_write, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            writer_img.writerow(folder_list)
            img_file_write.flush()
            
            self.printTxt('finished image retrieval\n')
        
        except Exception as e:
            print(e)
            self.printTxt('searching through directory failed\n')    
        
      
    def performTemplateMatching(self):
        
        try:
            #read parameters from entry
            plot_results = self.plot_results2.get()
            save_img = self.save_img.get()
            template_size_x = self.template_size_x.get()
            template_size_y = self.template_size_y.get()
            search_area_x = self.search_area_x.get()
            search_area_y = self.search_area_y.get()
            error_accpt = self.error_accpt.get()
            max_ptsToSkip = self.max_ptsToSkip.get()

        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)            
            self.printTxt('retrieving parameters failed\n')               
            
            
        failing = True
        while failing:  
            try:   
                directory_output = tkFileDialog.askdirectory(title='Output Directory') + '/'
                
                directory_img = tkFileDialog.askdirectory(title='Directory images for template matching') + '/'
                
                if self.save_img.get():
                    directory_output_img = tkFileDialog.askdirectory(title='Output directory images TM result')+ '/'
                
                img_start = tkFileDialog.askopenfilename(title='Open 1st template image', 
                                                         filetypes=[('Image file (*.jpg)', '*.jpg')],initialdir=os.getcwd())
                
                imgCooGCP_file_start = tkFileDialog.askopenfilename(title='Open GCP img coordinates of 1st template', 
                                                         filetypes=[('Txt file (*.txt)', '*.txt')],initialdir=os.getcwd())
                
                failing = False
                
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                self.printTxt('loading data for template matching failed, please try again\n')
        
        try:   
            #print parameters   
            self.printTxt('-----Parameter settings template matching-----\n')
            self.printTxt('-----Parameter settings template matching-----\n')
            self.printTxt('Template size x: ' + str(template_size_x) + '\n')
            self.printTxt('Template size y: ' + str(template_size_y) + '\n')
            self.printTxt('Search area size x: ' + str(search_area_x) + '\n')
            self.printTxt('Search area size y: ' + str(search_area_y) + '\n')
            self.printTxt('Error threshold: ' + str(error_accpt) + '\n')
            self.printTxt('Maximum points to skip: ' + str(max_ptsToSkip) + '\n')
            if save_img:   
                self.printTxt('Save images matching results\n')
            if plot_results:   
                self.printTxt('Plot results\n')     
 
 
            self.printTxt('-----start template matching-----\n')
 
            '''prepare master (template)'''
            #read image point coordinates of GCPs of template
            pts_table = pd.read_csv(imgCooGCP_file_start, header=None, index_col=False, delimiter='\t')
            pts_table = np.asarray(pts_table, dtype=np.float)
            img_pts = pts_table[:,1:3]
            
            pt_ids = pts_table[:,0]
            pt_ids = pt_ids.reshape(pt_ids.shape[0],1)
            
            #read template image 
            imgTemplate = cv2.imread(img_start, 0)     
            
            tmpMatch = tmp_match.templateMatch(template_size_x, template_size_y, search_area_x, search_area_y, plot_results)
            
            '''prepare quality control template matching'''
            img_pts_CooID = np.hstack((pt_ids, img_pts.reshape(img_pts.shape[0],2)))
            #calculate distances in image space between all points 
            pt_distances_template = np.asarray(tmpMatch.pt_distances(img_pts_CooID))            
            
            '''perform template matching for each image in dir'''
            for img_file in os.listdir(directory_img):                
                if '.jpg' in img_file:                    
                    self.printTxt('processing ' + img_file[:20] + '\n')
                    
                    #read search image 
                    imgSearch = cv2.imread(directory_img + img_file, 0)
            
                    #perform template matching
                    matched_points = tmpMatch.templateMatching(img_pts, imgTemplate, imgSearch)
                    try:
                        matched_points = np.hstack((pt_ids, matched_points.reshape(matched_points.shape[0],2)))
                    except Exception as e:
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)                        
                        continue
                    
                    pt_distances_search = np.asarray(tmpMatch.pt_distances(matched_points))
                    pt_distances_search = np.hstack((matched_points, pt_distances_search[:,1].reshape(pt_distances_search.shape[0],1)))
                    pt_distances_merged = np.hstack((pt_distances_search, pt_distances_template[:,1].reshape(pt_distances_template.shape[0],1)))
                    correct_matched_pts = pt_distances_merged[pt_distances_merged[:,3] < (pt_distances_merged[:,4] + pt_distances_merged[:,4] * error_accpt)]
                    correct_matched_pts = correct_matched_pts[correct_matched_pts[:,3] > (correct_matched_pts[:,4] - correct_matched_pts[:,4] * error_accpt)]
                    
                    if correct_matched_pts.shape[0] < matched_points.shape[0]:
                        self.printTxt('skipped ' + str(correct_matched_pts.shape[0] - matched_points.shape[0]) + ' points\n')
                    
                    
                    if (np.abs(correct_matched_pts.shape[0] - matched_points.shape[0])) <= max_ptsToSkip: 
                        #write output
                        write_file = open(directory_output + 'imgPtsGCP_' + img_file[:20] + '.txt', 'wb')
                        writer = csv.writer(write_file, delimiter='\t')
                        writer.writerows(correct_matched_pts[:,0:3])
                        write_file.flush()
                        write_file.close()
                                        
                        tmpMatch.plot_pts(imgSearch, correct_matched_pts[:,1:3], switchColRow=False, plt_title='', output_save=True,
                                          output_img=os.path.join(directory_output_img, 'templates_' + img_file[:20] + '.jpg'))
                    
                    else: 
                        tmpMatch.plot_pts(imgSearch, correct_matched_pts[:,1:3], switchColRow=False, plt_title='', output_save=True, 
                                          output_img=os.path.join(directory_output_img, 'templates_' + img_file[:20] + '.jpg'), edgecolor='red')
                        self.printTxt('no img coo of GCP\n')
            
            self.printTxt('-----finished template matching-----\n')
                        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            self.printTxt('template matching failed\n')
                            
        
    def getValuesFromTxtBox(self, parameter):
        try:
            paramRead = parameter.get()
            self.printTxt('Parameter set to: ' + str(paramRead) + '\n')
            
            return paramRead
        
        except Exception as e:
            print(e)
            self.printTxt('reading parameter failed, please try again\n')
  
  
    def loadData_images(self):

        failing = True
        while failing:
            directory_image_folders = tkFileDialog.askdirectory(title='Open Images Directory') + '/'
            
            try:
                '''search through directory for sub-directories'''
                if os.path.isdir(directory_image_folders):
                    for dirpath, dirsubpaths, dirfiles in os.walk(directory_image_folders):
                        if len(dirsubpaths) >= 1:
                            folders = dirsubpaths
                            break
                        else:
                            self.printTxt('empty directory: ' + dirpath + '\n')
                            sys.exit()
                else: 
                    self.printTxt('directory ' + directory_image_folders + ' not found\n')
                    sys.exit()
                    
                '''search through sub-directories for master file'''
                masterImg_list = []
                for video_folder in folders:
                    directory_video = directory_image_folders + video_folder + '/' + video_folder + '/'
                    for video_file in os.listdir(directory_video):
                        if '_0.jpg' in video_file:
                            masterImg_list.append([directory_video, video_file])
                masterImg_list = sorted(masterImg_list, key=lambda video: (video[0], video[1]))
                        
                '''search through sub-directories for image files'''
                image_all_list = []
                for image_folder in folders:
                    directory_image = directory_image_folders + image_folder + '/' + image_folder + '/'
                    for image_file in os.listdir(directory_image):
                        if 'jpg' in image_file or 'png' in image_file:
                            image_all_list.append([directory_image, image_file])
                image_all_list = sorted(image_all_list, key=lambda image: (image[0], image[1]))
                
                self.printTxt(str(len(image_all_list)) + ' images are loaded\n')
                
                failing = False
                
                return image_all_list, masterImg_list
            
            except Exception as e:
                print(e)
                self.printTxt('reading images failed, please try again\n')

            
    def loadData_waterlineApprox(self):      
        
        failing = True
        while failing:
            '''read water line approx files'''
            try:
                if self.waterline_approx_steady.get():
                    
                    #read file
                    waterlines_approx = tkFileDialog.askopenfilename(title='Open waterline approximated', filetypes=[('Txt files (*.txt)',
                                                                                                                      '*.txt')],initialdir=os.getcwd())
                    
                    self.printTxt(waterlines_approx + ' water line file loaded\n')
            
                    return waterlines_approx
                
                else:
                    #read directory
                    directory_waterline_approx = tkFileDialog.askdirectory(title='Open waterlines approximated directory') + '/'
                    
                    #get files in directory
                    waterlines_approx = []
                    for waterline_approx_file in os.listdir(directory_waterline_approx):
                        if '_waterline_approx.txt' in waterline_approx_file:
                            waterlines_approx.append([directory_waterline_approx, waterline_approx_file])
                    waterlines_approx = sorted(waterlines_approx)
        
                    self.printTxt(str(len(waterlines_approx)) +  ' waterlines (approximated) loaded\n')
        
                    failing = True
        
                    return waterlines_approx
                
            except Exception as e:
                print(e)
                self.printTxt('reading waterlines approximated failed, please try again\n')            
            
               
    def waterlineDetection(self):
        
        kp_nbr = 3000
        nbr_good_matches = 40        

        try:
            #get values from textbox entry
            bilat_kernelStr = self.bilat_kernelStr.get()
            bilat_kernel = bilat_kernelStr.split(':')
            bilat_kernel = [int(bilat_kernel[0]), int(bilat_kernel[1])]                    
            if self.watersideInt.get() == 0:
                waterside = "right"
            else:
                waterside = "left"                         
            buff_size_waterlineSearch = self.buff_size_waterlineSearch.get()            
            thresh_hist_waterlineSearch = self.thresh_hist_waterlineSearch.get()            
            add_thresh_grey = self.add_thresh_grey.get()              
            add_thresh_tempText = self.add_thresh_tempText.get()
            canny_kernel = self.canny_kernel.get()            
            nan_clip_size = self.nan_clip_size.get()            
            use_canny = self.use_canny.get()            
            plot_results = self.plot_results.get()
            perform_coregist = self.perform_coregist.get()
            do_continue = self.do_continue.get()
            if do_continue:
                video_value = self.video_value.get()
             
            #print parameters   
            self.printTxt('-----Parameter settings water line detection-----\n')
            self.printTxt('Buffer size: ' + str(buff_size_waterlineSearch) + '\n')
            self.printTxt('Threshold histogram: ' + str(thresh_hist_waterlineSearch) + '\n')
            self.printTxt('Add value to grey value thresh: ' + str(add_thresh_grey) + '\n')
            self.printTxt('Add value to temp texture thresh: ' + str(add_thresh_tempText) + '\n')
            self.printTxt('Kernel size bilateral filter: ' + str(bilat_kernel) + '\n')
            self.printTxt('Kernel size canny filter: ' + str(canny_kernel) + '\n')
            self.printTxt('Clip size NaN values: ' + str(nan_clip_size) + '\n')
            self.printTxt('Waterside: ' + waterside + '\n')
            if do_continue:            
                self.printTxt('Start at image: ' + video_value + '\n')            
            if self.waterline_approx_steady.get():
                self.printTxt('Waterline approx: single file\n')
            if use_canny:   
                self.printTxt('Use Canny filter\n')
            if plot_results:   
                self.printTxt('Plot results\n')            
            if perform_coregist:   
                self.printTxt('Perform co-registration\n')                 
            
            
            #load images and waterlines           
            image_all_list, masterImg_list = self.loadData_images()
            waterlines_approx = self.loadData_waterlineApprox()           

            #set output location
            failing = True
            while failing:
                try:
                    directory_results = tkFileDialog.askdirectory(title='Open Output Directory') + '/'
                    failing = False
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)                       
                    print(e)
                    self.printTxt('reading output directory failed, please try again\n')
            
            self.printTxt('Output Directory: ' + directory_results + '\n')
            self.dummy()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)   
            print(e)
            self.printTxt('water line detection failed, please try again\n')
            
        try:
            #prepare logfile
            logfile = open(directory_results + 'logfile_.txt', 'wb')
            writer = csv.writer(logfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['water line detection'])
            writer.writerow(['buffer size'] + [str(buff_size_waterlineSearch)])
            writer.writerow(['correlation threshold'] + [str(thresh_hist_waterlineSearch)])
            writer.writerow(['---------------------------------------------------'])
            logfile.flush()
                       
            '''------------------------------------------------------------------------------------------------------------------------------------------
            start processing
            ---------------------------------------------------------------------------------------------------------------------------------------------'''            
            #processing per master image file (and thus hour)
            coreg = wl_func.Coregsitration() 
            imgProc = wl_func.ImageProcess()
            wlEstim = wl_func.WaterlineEstimation()
            
            only_once = False
            print(masterImg_list)
            for masterImg in masterImg_list:
                if only_once:
                    continue

                if self.waterline_single.get():
                    only_once = True
                     
                if do_continue: 
                    if str((masterImg[1])[:-6]) == video_value:
                        do_continue = False
                    if do_continue:
                        continue               
            
                try:
                    self.printTxt('---------------------------------------------------------------------------\n')
                    self.printTxt('processing: ' + masterImg[0] + ' ' + masterImg[1] + '\n')
                    self.printTxt('---------------------------------------------------------------------------\n')
                    self.dummy()
                    
                    masterImg_name = (masterImg[1])[11:-6]
                    
                    temporary_folder = masterImg[0] + 'temp_' + masterImg_name + '/'
                    if not os.path.exists(temporary_folder):     #open directory for temporary saving of video frames
                        os.makedirs(temporary_folder)  
                    
                    output_folder = directory_results + masterImg[1][0:-6] + '/'
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    output_dir_coreg = output_folder + 'coreg/'
                    if not os.path.exists(output_dir_coreg):
                        os.makedirs(output_dir_coreg)
                        perform_coregist = True
                    else:
                        perform_coregist = False
                    
                    output_results_dir =  output_dir_coreg + 'results/'
                    if not os.path.exists(output_results_dir):
                        os.makedirs(output_results_dir)      
                                           
                    '''----------------------------------------------------------------------------------------------------'''   
                    '''get images from corresponding master image (and thus time/hour)'''
                    image_list = []
                    for image in image_all_list:
                        if masterImg[1][:-6] == image[1][:-6]:
                            if masterImg_name in image[1]:
                                image_list.append(image)
                                
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)   
                    print(e)
                    print('data preparation failed')
                    continue
                                    
           
                '''----------------------------------------------------------------------------------------------------'''   
                '''co-register images of varying sources'''
                try:
                    if perform_coregist:     
                        #co-register images and frames                 
                        coreg.coregistration(image_list, output_dir_coreg, kp_nbr, False, 
                                             True, nbr_good_matches, 
                                             False, True, False)  #[video_list[0][0],  (video_list[0][1])[0:-5] + '_0.jpg']       
                        
                        #read co-registered images in directory into image list                                        
                        print(masterImg[1][0:5])
                        
                        image_list = wl_func.read_files_in_dir(output_dir_coreg, masterImg[1] + ' images coreg', masterImg[1][0:5])
                    
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)   
                    print(e)
                    print('coregistration failed')
                    continue
              
                '''----------------------------------------------------------------------------------------------------'''   
                '''start WaterLineDetection'''
                try:        
                    #calculate and filter temporal texture        
                    border_mask_file = os.path.join(output_dir_coreg, 'mask_border.txt')
                    mean_stills, tempText_stills  = imgProc.tempTexture(image_list, output_results_dir, border_mask_file, False)
                    
                    #estimate waterline
                    if not self.waterline_approx_steady.get():
                        waterlineApproxRead = None
                        for waterline_approx_file in waterlines_approx:
                            if waterline_approx_file[1][:-21] == masterImg[1][:-6]:        
                                waterlineApproxRead = waterline_approx_file[0] + '/' + waterline_approx_file[1]
                        if waterlineApproxRead == None:
                            print('no waterline approximatio given')
                            continue
                                
                    waterline_stills, BB, corr_hist = wlEstim.waterline_estimate(tempText_stills, mean_stills, output_results_dir, waterlineApproxRead, 
                                                                                 plot_results, thresh_hist_waterlineSearch, buff_size_waterlineSearch, waterside, 
                                                                                 add_thresh_tempText, add_thresh_grey, bilat_kernel, canny_kernel, use_canny, nan_clip_size)
                        
                    print('waterline tool executed')
                
                except Exception as e:
                    
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    del exc_type, exc_obj

                    cv2.imwrite(directory_results + 'waterline_' + (masterImg[1])[:-6] + '.jpg', mean_stills)
                    
                    print(e, exc_tb.tb_lineno)
                    print('waterline detection failed')
                    continue
            
                
                '''----------------------------------------------------------------------------------------------------'''   
                '''create new waterline approximation'''
                try:
                    #log report
                    log_file_temp = open(output_results_dir + 'logfile.txt', 'rb')
                    reader = csv.reader(log_file_temp, delimiter=' ', quoting=csv.QUOTE_MINIMAL)        
                    writer.writerow(['___________________'])
                    writer.writerow([str((masterImg[1])[:-6])])
                    for row in reader:
                        writer.writerows([row])
                    logfile.flush()
                    log_file_temp.close()                         
                    
                    '''----------------------------------------------------------------------------------------------------'''   
                    '''save results'''                                  
                    #copy to result folder:
                    os.system('cp -r -f ' + output_results_dir + 'wasserlinie.txt ' + directory_results)
                    os.system('mv '+ directory_results + 'wasserlinie.txt ' + directory_results + 'wasserlinie_' + (masterImg[1])[:-5] + '.txt')  
                    os.system('cp -r -f ' + output_results_dir + 'waterline.jpg ' + directory_results)
                    os.system('mv '+ directory_results + 'waterline.jpg ' + directory_results + 'waterline_' + (masterImg[1])[:-5] + '.jpg')
                                          
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)   
                    print(e)
                    print('waterline approximation failed') 
            
            
                #remove temporary directory of video frames and waterline
                os.system('rm -r -f ' + temporary_folder)           
            
            logfile.close()
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)   
            print(e)
            

def main():        

    print('openCV version: ' + cv2.__version__)
    
    root = Tk()
    
    app = WaterlineTool(root)   
    
    root.mainloop()
    
    
    
    root.destroy() # optional; see description below        



if __name__ == "__main__":
    main()  