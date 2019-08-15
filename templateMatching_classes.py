# Copyright (c) 2018, Anette Eltner
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pylab as plt
import cv2


#perform template matching in images to detect GCPs
class templateMatch:
    
    #define template position and size and search area position
    def __init__(self, template_size_x=300, template_size_y=300,
                 search_area_x=0, search_area_y=0, plot_results=False):
        self.template_size_x = template_size_x
        self.template_size_y = template_size_y
        self.search_area_x = search_area_x
        self.search_area_y = search_area_y
        self.plot_results=plot_results
    
    
    #define template at image point position (of corresponding GCP)
    def getTemplateAtImgpoint(self, img, img_pts, template_width=10, template_height=10):
    #consideration that row is y and column is x   
    #careful that template extent even to symmetric size around point of interest 
        
        template_img = []
        anchor_pts = []
        for pt in img_pts:
            if img_pts.shape[1] > 2:
                template_width_for_cut_left = pt[2]/2
                template_width_for_cut_right = pt[2]/2 + 1
            elif template_width > 0:
                template_width_for_cut_left = template_width/2
                template_width_for_cut_right = template_width/2 + 1
            else:
                print 'missing template size assignment'
            
            if img_pts.shape[1] > 2:
                template_height_for_cut_lower = pt[3]/2
                template_height_for_cut_upper = pt[3]/2 + 1
            elif template_height > 0:
                template_height_for_cut_lower = template_height/2
                template_height_for_cut_upper = template_height/2 + 1
            else:
                print 'missing template size assignment'
            
            cut_anchor_x = pt[0] - template_width_for_cut_left
            cut_anchor_y = pt[1] - template_height_for_cut_lower
            
            #consideration of reaching of image boarders (cutting of templates)
            if pt[1] + template_height_for_cut_upper > img.shape[0]:
                template_height_for_cut_upper = np.int(img.shape[0] - pt[1])
            if pt[1] - template_height_for_cut_lower < 0:
                template_height_for_cut_lower = np.int(pt[1])
                cut_anchor_y = 0
            if pt[0] + template_width_for_cut_right > img.shape[1]:
                template_width_for_cut_right = np.int(img.shape[1] - pt[0])
            if pt[0] - template_width_for_cut_left < 0:
                template_width_for_cut_left = np.int(pt[0])
                cut_anchor_x = 0
            
            template = img[pt[1]-template_height_for_cut_lower:pt[1]+template_height_for_cut_upper, 
                           pt[0]-template_width_for_cut_left:pt[0]+template_width_for_cut_right]
            
            #template_img = np.dstack((template_img, template))
            template_img.append(template)
            
            anchor_pts.append([cut_anchor_x, cut_anchor_y])
            
        anchor_pts = np.asarray(anchor_pts, dtype=np.float32) 
        #template_img = np.delete(template_img, 0, axis=2) 
        
        return template_img, anchor_pts #anchor_pts defines position of lower left of template in image
    
    
    #template matching for automatic detection of image coordinates of GCPs
    def performTemplateMatch(self, img_extracts, template_img, anchor_pts):
        new_img_pts = []
        template_nbr = 0
        
        count_pts = 0
        while template_nbr < len(template_img):
            template_array = np.asarray(template_img[template_nbr])
            if (type(img_extracts) is list and len(img_extracts) > 1) or (type(img_extracts) is tuple and len(img_extracts.shape) > 2):      
                img_extract = img_extracts[template_nbr]
            else:
                img_extract = img_extracts
            res = cv2.matchTemplate(img_extract, template_array, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #min_loc for TM_SQDIFF
            match_position_x = max_loc[0] + template_array.shape[1]/2
            match_position_y = max_loc[1] + template_array.shape[0]/2
            del min_val, min_loc
            
            if max_val > 0.9:
                new_img_pts.append([match_position_x + anchor_pts[template_nbr,0], 
                                    match_position_y + anchor_pts[template_nbr,1]])
                count_pts = count_pts + 1
                 
            template_nbr = template_nbr + 1
    
            if self.plot_results:    
                plt.subplot(131),plt.imshow(res,cmap = 'gray')
                plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                plt.plot(match_position_x-template_array.shape[1]/2, match_position_y-template_array.shape[0]/2, "r.", markersize=10)
                plt.subplot(132),plt.imshow(img_extract,cmap = 'gray')
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])    
                plt.plot(match_position_x, match_position_y, "r.", markersize=10)
                plt.subplot(133),plt.imshow(template_array,cmap = 'gray')
                plt.title('Template'), plt.xticks([]), plt.yticks([])
                plt.show()
            
        new_img_pts = np.asarray(new_img_pts, dtype=np.float32)
        new_img_pts = new_img_pts.reshape(count_pts, 2)
             
        return new_img_pts
    
    
    #perform template matching
    def templateMatching(self, img_pts, img_for_template, img_for_search):
        
        #if no search area defined, squared search are triple of template size    
        if self.search_area_x == 0 and self.search_area_y == 0:
            search_area = np.ones((img_pts.shape[0], 2)) * self.template_size_x * 3
        #search_area_...: if only in x direction defined, used as same size for y (squared search area)
        elif self.search_area_x > 0 and self.search_area_y == 0:
            search_area = np.ones((img_pts.shape[0], 2)) * self.search_area_x
        elif self.search_area_x == 0 and self.search_area_y > 0:
            search_area = np.ones((img_pts.shape[0], 2)) * self.search_area_y
        else:
            search_area_x = np.ones((img_pts.shape[0], 1)) * self.search_area_x
            search_area_y = np.ones((img_pts.shape[0], 1)) * self.search_area_y
            search_area = np.hstack((search_area_x.reshape(search_area_x.shape[0],1), 
                                     search_area_y.reshape(search_area_y.shape[0],1)))
        
        #if template size defined only in one direction, squared template size are calculated    
        if self.template_size_x == 30 and self.template_size_y == 30:
            template_sizes = np.ones((img_pts.shape[0], 2)) * self.template_size_x
        #else side specific template size
        else:
            template_size_x = np.ones((img_pts.shape[0], 1)) * self.template_size_x
            template_size_y = np.ones((img_pts.shape[0], 1)) * self.template_size_y
            template_sizes = np.hstack((template_size_x.reshape(template_size_x.shape[0],1), 
                                     template_size_y.reshape(template_size_y.shape[0],1)))
        
        #calculate template with corresponding template size
        template_prepare = np.hstack((img_pts, template_sizes))
        pt_templates, _ = self.getTemplateAtImgpoint(img_for_template, template_prepare)
            
                
        #approximation of template position for subsequent images
        searchArea_perPoint = np.hstack((img_pts, search_area))    
        searchArea_clip, anchor_pts = self.getTemplateAtImgpoint(img_for_search, searchArea_perPoint)
        
        #perform template matching
        img_pts_matched = self.performTemplateMatch(searchArea_clip, pt_templates, anchor_pts)
        
        return img_pts_matched


    #quality measure of detected templates considering distance pattern
    def pt_distances(self, pts):
    #pts: id and x y coordinates of image points (np array)
    
        pt_distances_sum = []
    
        for pt in pts:
            
            pt_for_dist = np.ones((pts.shape[0], pts.shape[1])) * pt 
            pt_dist = np.sqrt(np.square(pt_for_dist[:,1] - pts[:,1]) + np.square(pt_for_dist[:,2] - pts[:,2]))
            
            pt_distances_sum.append([pt[0], np.sum(pt_dist)])
            
        return pt_distances_sum
    
    
    #plot matched points into corresponding image
    def plot_pts(self, img, points, switchColRow=False, plt_title='', output_save=False, output_img=None,
                 edgecolor='blue'):
        plt.clf()
        plt.figure(frameon=False)
        plt.gray()
        if switchColRow:
            plt.plot([p[1] for p in points],
                    [p[0] for p in points],
                    marker='o', ms=5, color='none', markeredgecolor=edgecolor, markeredgewidth=1)
        else:
            plt.plot([p[0] for p in points],
                     [p[1] for p in points],
                     marker='o', ms=5, color='none', markeredgecolor=edgecolor, markeredgewidth=1)
        plt.title(plt_title)
        plt.axis('off')
        plt.imshow(img)
        
        if not output_save:
            plt.waitforbuttonpress()
            plt.close()
        else:
            plt.savefig(output_img,  dpi=600)

