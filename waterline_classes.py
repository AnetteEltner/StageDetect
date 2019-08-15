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

# encoding=utf8

import os, sys, csv
import cv2
import numpy as np
import pylab as plt
import pandas as pd

import skimage.morphology
from skimage.morphology import square
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.filters.rank import mean_bilateral

from PIL import Image, ImageDraw

import statsmodels.nonparametric.smoothers_lowess
from scipy import stats

from shapely.geometry import LineString, Point#, Polygon

import seaborn as sns
import sklearn.cluster as cluster

from __builtin__ import True


import draw_classes as drw
import cv_classes as cv_cl


'''read all files in folder into a list'''
def read_files_in_dir(folder, file_name, filter_val=None):
    '''read files in folder into list'''
    frame_list = []
    for frame_file in os.listdir(folder):
        if filter_val == None:
            frame_list.append([folder, frame_file])
        elif filter_val in frame_file:
            frame_list.append([folder, frame_file])
    
    if len(frame_list) < 1:
        print('no frames of ' + file_name + ' found.')
        sys.exit()
    else:
        print(str(len(frame_list)) + ' frames are read for ' + file_name)
    
    frame_list = sorted(frame_list, key=lambda frame: frame[0])        

    return frame_list   #list of frames conaining directory and frame name

'''calculate moving average for specified number of samples'''
def movingaverage(self, mylist, N):
    cumsum, moving_aves = [0], []
    
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
    
    return moving_aves

'''convert x,y values into dataframe to drop duplicates and return cleaned array'''
def with_dataframe(x, y, int_out=False, row_wise=False):
    df = pd.DataFrame({'x':x, 'y':y})
    
    if row_wise:
        df_y_drop = df.drop_duplicates()
    else:
        df_x_drop = df.drop_duplicates(subset=['x'])
        df_y_drop = df_x_drop.drop_duplicates(subset=['y'])    
    
    if int_out:
        df_y_drop = np.asarray(df_y_drop, dtype=np.int)
    else:
        df_y_drop = np.asarray(df_y_drop)     
    
    return df_y_drop

'''RANSAC to filter outliers for various models'''
class Ransac:
    def __init__(self):
        pass
    
    #perform outlier detection
    def ransac(self, data, model_class, min_samples, residual_threshold,
               is_data_valid=None, is_model_valid=None,
               max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
               poly_order=1):

    
        best_model = None
        best_inlier_num = 0
        best_inlier_residuals_sum = np.inf
        best_inliers = None
    
        if not isinstance(data, list) and not isinstance(data, tuple):
            data = [data]
    
        # make sure data is list and not tuple, so it can be modified below
        data = list(data)
        # number of samples
        N = data[0].shape[0]
    
        for _ in range(max_trials):
    
            # choose random sample set
            samples = []
            random_idxs = np.random.randint(0, N, min_samples)
            for d in data:
                samples.append(d[random_idxs])
            samples = np.hstack(np.array(samples))
    
            # check if random sample set is valid
            if is_data_valid is not None and not is_data_valid(*samples):
                continue
    
            # estimate model for current random sample set
            sample_model = model_class()
            if poly_order > 1:
                sample_model.estimate(samples, poly_order)
            else:
                sample_model.estimate(samples)
    
            # check if estimated model is valid
            if is_model_valid is not None and not is_model_valid(sample_model,
                                                                 *samples):
                continue
    
            sample_model_residuals = np.abs(sample_model.residuals(*data))
            # consensus set / inliers
            sample_model_inliers = sample_model_residuals < residual_threshold
            sample_model_residuals_sum = np.sum(sample_model_residuals**2)
    
            # choose as new best model if number of inliers is maximal
            sample_inlier_num = np.sum(sample_model_inliers)
            if (
                # more inliers
                sample_inlier_num > best_inlier_num
                # same number of inliers but less "error" in terms of residuals
                or (sample_inlier_num == best_inlier_num
                    and sample_model_residuals_sum < best_inlier_residuals_sum)
            ):
                best_model = sample_model
                best_inlier_num = sample_inlier_num
                best_inlier_residuals_sum = sample_model_residuals_sum
                best_inliers = sample_model_inliers
                if (
                    best_inlier_num >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum
                ):
                    break
    
        # estimate final model using all inliers
        if best_inliers is not None:
            # select inliers for each data array
            for i in range(len(data)):
                data[i] = data[i][best_inliers]
            data = np.hstack(np.array(data))
            if poly_order > 1:
                best_model.estimate(data, poly_order)
            else:
                best_model.estimate(data)
        
        return best_model, best_inliers

'''linear model for RANSAC class'''
class LinearModel_ransac:
    """linear system solved using linear least squares"""

    def estimate(self, data):
        x = data[:, 0]
        y = data[:, 1]
        
        #fit line
        params = stats.linregress(x,y)
        
        self._params = params
        
        return params
    
    def residuals(self, data):
        params = self._params

        x = data[:, 0]
        y = data[:, 1]
        
        x_fit = np.array(sorted(x))
        y_fit = params[0]*x_fit + params[1]
        
        #estimate shortest distance to line
        line = LineString(np.column_stack([x_fit, y_fit]))
        residuals = []
        for pt in np.column_stack([x, y]):
            point = Point(pt)
            residuals.append(point.distance(line))
        
        return residuals
    
    def predict_y(self, x):
        params = self._params
        
        y = params[0]*x + params[1]

        return y
    
    
'''clean final water line for outliers using linear regression'''    
class LineAndPointOperation:
    def __init__(self):
        pass
        
    def rotate_smooth_rotateback(self, xy, plot_results=False):
        '''linear regression to find line for points with RANSAC and following least square fit
            afterwards rotation around main slope'''
    
        '''fit line to points'''
        xy_sw = np.empty((xy.shape[0], xy.shape[1]))
        xy_sw[:,0] = xy[:,1]
        xy_sw[:,1] = xy[:,0]
        xy = xy_sw
        
        #ransac approach to find first line
        model = LinearModel_ransac()
        iteration = 0
        model_robust = None
        while iteration <= 10: #repeat until successful line parameters estimated
            iteration = iteration + 1
            try:
                rc = Ransac()
                model_robust, inliers = rc.ransac(xy, LinearModel_ransac, min_samples=xy.shape[0]/2,
                                                  residual_threshold=0.0075, max_trials=10, stop_sample_num=xy.shape[0]*0.9,
                                                  stop_residuals_sum=0.01)
                
                parameters = model.estimate(np.column_stack([xy[inliers, 0], xy[inliers, 1]]))
                if not np.isnan(parameters[0]): 
                    break
                else:
                    continue
            except:
                continue
    
        del model_robust
    
        xy_sort = np.asarray(sorted(xy, key=lambda xord: xord[0]))
        x_sorted = xy_sort[:,0]
        y_sorted = xy_sort[:,1]
        
        #using least square fit to find final line
        m, c = np.polyfit(x_sorted, y_sorted, 1)
    
    
        '''rotation of points'''
        rotAngle = np.arctan(m)
        #print (rotAngle * (180/(np.pi)))
        rotMat = np.asarray([[np.cos(rotAngle), -1 * np.sin(rotAngle)], [np.sin(rotAngle), np.cos(rotAngle)]]).T
        
        #centre_point = np.asarray([np.sum(x_sorted)/x_sorted.shape[0], np.sum(y_sorted)/x_sorted.shape[0]])
        centre_point = xy_sort[0,:]
        xy_shift = np.mat(xy_sort).T - np.mat(np.ones((xy_sort.shape[0],2)) * centre_point).conj().T 
        xy_sort_rot = np.mat(rotMat) * np.mat(xy_shift).conj() + np.mat(np.ones((xy_sort.shape[0],2)) * centre_point).conj().T 
        xy_sort_rot = np.asarray(xy_sort_rot.T)   
        
        
        '''smooth line'''
        xy_final_rot = np.empty((xy_sort_rot.shape[0], xy_sort_rot.shape[1]))
        xy_final_rot[:,0] =  xy_sort_rot[:,1]
        xy_final_rot[:,1] =  xy_sort_rot[:,0]
        xy_sort_rot_smooth = statsmodels.nonparametric.smoothers_lowess.lowess(xy_final_rot[:,0], xy_final_rot[:,1], frac=.2)
        
        
        '''rotate back'''
        rotAngle_back = -1*rotAngle
        #print (rotAngle_back * (180/(np.pi)))
        rotMat_inv = np.asarray([[np.cos(rotAngle_back), -1 * np.sin(rotAngle_back)], [np.sin(rotAngle_back), np.cos(rotAngle_back)]]).T
        xy_shift_back = np.mat(xy_sort_rot_smooth).T - np.mat(np.ones((xy_sort_rot_smooth.shape[0],2)) * centre_point).conj().T
        xy_sort_smooth = np.mat(rotMat_inv) * np.mat(xy_shift_back).conj() + np.mat(np.ones((xy_sort_rot_smooth.shape[0],2)) * centre_point).conj().T 
        xy_sort_smooth = np.asarray(xy_sort_smooth.T)   
    
    
        if plot_results:
            y_fit_rans = parameters[0] * x_sorted + parameters[1]
         
            y_fit_lsq = m * x_sorted + c
            
            fig1 = plt.figure()
            ax = fig1.add_subplot(111, aspect='equal')
            ax.plot(xy[:,0], xy[:,1], '.k', markersize=3)
            ax.plot(x_sorted, y_fit_rans, '-g', linewidth=1)
            ax.plot(x_sorted, y_fit_lsq, '-b', linewidth=1)
            #ax.plot(xy_sort_rot_smooth[:,0], xy_sort_rot_smooth[:,1], '.r', markersize=3)
            ax.plot(xy_sort_smooth[:,0], xy_sort_smooth[:,1], '.r', markersize=3)
            plt.show()
            
        return xy_sort_smooth
        

    '''find centroid of points'''
    def centeroidnp(self, arr):
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length
    
    
    '''find clusters using sklearn library'''    
    def analyse_for_clusters(self, data, algorithm, args, kwds, ref_line, ref_side, len_shift_to_ref_side, 
                             plot_cluster=False, rg_grey=False):
        plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
        thresh_cluster = 15
        
        labels = algorithm(*args, **kwds).fit_predict(data)
        clustered_points = np.hstack((labels.reshape(labels.shape[0],1), data))
        
        '''check if more than one cluster detected'''
        unique_cluster_nbr = np.unique(clustered_points[:,0])
        if unique_cluster_nbr.shape[0] > 1:  
            centroid_clusters = []
            centroid_cluster_fail_safe = []
            for cluster_nbr in unique_cluster_nbr:
                point_cluster = clustered_points[clustered_points[:,0]==cluster_nbr]            
                centroid_cluster = self.centeroidnp(point_cluster[:,1:3])
                centroid_cluster_fail_safe.append([cluster_nbr, centroid_cluster])
                
                '''check if points are clumped (ratio std to average of distances to centroid has to be above threshold)''' 
                dist_cluster_to_centroid = []
                for point_cl in point_cluster:
                    point_circled = Point(point_cl[1:3])
                    dist_pt = point_circled.distance(Point(centroid_cluster))
                    dist_cluster_to_centroid.append(dist_pt)
                dist_cluster_to_centroid = np.asarray(dist_cluster_to_centroid)
                # print np.average(dist_cluster_to_centroid) + 3*np.std(dist_cluster_to_centroid)
                if np.average(dist_cluster_to_centroid) + 3*np.std(dist_cluster_to_centroid) < thresh_cluster:
                    continue
                
                centroid_clusters.append([cluster_nbr, centroid_cluster])
            
            if len(centroid_clusters) <= 1:
                centroid_clusters = centroid_cluster_fail_safe
            
            '''choose cluster at land side'''
            waterline_approx_line = LineString(ref_line)
            #get line far at land side
            if ref_side == 'right' and rg_grey:
                waterline_offset_land = waterline_approx_line.parallel_offset(len_shift_to_ref_side, 'right')
            elif ref_side == 'left' and rg_grey:    
                waterline_offset_land = waterline_approx_line.parallel_offset(len_shift_to_ref_side, 'left')
            elif ref_side == 'right' and not rg_grey:
                waterline_offset_land = waterline_approx_line.parallel_offset(len_shift_to_ref_side, 'left')
            elif ref_side == 'left' and not rg_grey:    
                waterline_offset_land = waterline_approx_line.parallel_offset(len_shift_to_ref_side, 'right')        
            waterline_offset_land_centroid = waterline_offset_land.centroid
            
            '''find distance of closest point of line to centroid'''
            shortest_dist = 3*len_shift_to_ref_side
            for centroid in centroid_clusters:
                centroid_pt = Point(np.asarray(centroid[1]))
                #dist_nearest_pt = waterline_offset_land.project(centroid_pt)
                dist_nearest_pt = centroid_pt.distance(waterline_offset_land_centroid)
                if shortest_dist > dist_nearest_pt:
                    shortest_dist = dist_nearest_pt
                    id_clust_closest_land = centroid[0]
            waterline = clustered_points[clustered_points[:,0]==id_clust_closest_land]
            waterline = waterline[:,1:3]   
    
            if plot_cluster:
                palette = sns.color_palette('deep', np.unique(labels).max() + 1)
                colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
                plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)    #data.T[0], data.T[1]
                frame = plt.gca()
                frame.axes.get_xaxis().set_visible(False)
                frame.axes.get_yaxis().set_visible(False)
                plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
                
                return plt, waterline
            
            else:
                return None, waterline
            
        else:
            return None, data
        

    '''detect contours of region growing and select largest boundary'''
    def detect_contours_RG(self, result_regionGrown, mask_border):    
    #     ring_gap_tresh = 30
        thresh_second_contour_len = 0.25 #ratio to first contour
        
        (cnts, _) = cv2.findContours(result_regionGrown, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(cnts, key = cv2.contourArea, reverse = True)
    
        '''check all contours and keep longest considering already bounding box'''
        contour_1st = True
        waterline = np.empty(1)
        i = -1
        i_out = None
    #     not_use_this_contour = False
        for contour in contours:
            i = i + 1
            
            contour = [item for sublist in contour for item in sublist]
            len_cnts = len(contour)
            contour = np.array(contour).reshape(len_cnts, 2)
            
            #remove contour bounding box
            contour_clean = []
            count_pts = 0
            for point_cont in contour:
                if mask_border[point_cont[1], [point_cont[0]]] == False:
                    contour_clean.append(point_cont)
                    count_pts = count_pts + 1
            contour_clean = np.asarray(contour_clean)
                        
            if not contour_clean.shape[0] > 1:
                continue     
    
            contourLS_prior = LineString(contour_clean)
            if contour_1st:
                contourLS_posterior = contourLS_prior
                contour_1st = False
                waterline = contour_clean
                i_out = i
                continue
            
            #keep only longest contour
            if contourLS_posterior.length < contourLS_prior.length:
                contourLS_posterior = contourLS_prior
                waterline = contour_clean
                i_out = i
        
        ''''find second longest contour (considering bounding box)'''
        if not i_out == None and len(contours) > 1:
            contour_1st = True
            waterline_2 = np.empty(1)
            i = -1
    #         not_use_this_contour = False
            for contour in contours:
                i = i + 1
                
                #skip previously detected longest line
                if i == i_out:
                    continue
                
                contour = [item for sublist in contour for item in sublist]
                len_cnts = len(contour)
                contour = np.array(contour).reshape(len_cnts, 2)            
        
                #remove contour bounding box
                contour_clean = []
                for point_cont in contour:
                    if mask_border[point_cont[1], [point_cont[0]]] == False:
                        contour_clean.append(point_cont)
                contour_clean = np.asarray(contour_clean)
                            
                if not contour_clean.shape[0] > 1:
                    continue
                
                contourLS_prior = LineString(contour_clean)
                if contour_1st:
                    contourLS_posterior = contourLS_prior
                    contour_1st = False
                    waterline_2 = contour_clean
                    continue
                
                #keep only longest contour
                if contourLS_posterior.length < contourLS_prior.length:
                    contourLS_posterior = contourLS_prior
                    waterline_2 = contour_clean
            
            waterline_1_LS = LineString(waterline)
            if waterline_2.shape[0] <= 1:
                return waterline
            else:
                waterline_2_LS = LineString(waterline_2)        
                if waterline_2_LS.length > thresh_second_contour_len * waterline_1_LS.length:
                    waterline = np.vstack((waterline, waterline_2))
                    
        return waterline
           
    
    '''detect contours of regions found via Canny and select largest boundary'''
    def detect_contours_Canny(self, Img_canny_binary, waterline_approx_len, mask_border): 
        (cnts, _) = cv2.findContours(Img_canny_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(cnts, key = cv2.contourArea, reverse = True)
        contour_1st = True
               
        #print(contours)
        
        canny_min_len = 3
        
        ''''iterate if all detected edges too short to be selected, if so decrease minimum edge length'''
        iteration = 0
        while iteration < 5:
            for contour in contours:
                contour = [item for sublist in contour for item in sublist]
                len_cnts = len(contour)
                contour = np.array(contour).reshape(len_cnts, 2)
                #contours_canny_final = contour
                
                if not contour.shape[0] > 1:
                    continue
                
                #contour has to have minimum length (fraction of approximated line) to be included as contour of waterline 
                contourLS = LineString(contour)
                if contourLS.length > (waterline_approx_len / canny_min_len):
                    if contour_1st:
                        contours_canny_final = contour
                        contour_1st = False
                    else:
                        contours_canny_final = np.vstack((contours_canny_final, contour))             
    
            '''remove bounding box and keep only contour in middle (erosion)'''
            waterline = []
            for point_cont in contours_canny_final:
                if mask_border[point_cont[1], [point_cont[0]]] == False:
                    waterline.append(point_cont)
            waterline = np.asarray(waterline)
            
            
            if not waterline.shape[0] > 5:  #if not enough edges found repeat (minimum number of edge vertices)
                iteration = iteration + 1
                canny_min_len = canny_min_len + 0.5
                continue
            else:
                break
                
        return waterline
        

'''co-register images of a sequence with openCV algorithms'''      
class Coregsitration:
    
    def __init__(self):
        pass
    
    '''perform coregistration'''
    def coregistration(self, image_list, directory_out, kp_nbr=None, sift_vers=False, 
                       feature_match_twosided=False, nbr_good_matches=10, 
                       visualize=False, master_0 = True, frames=False, master_frame=None):
        #image_list: list of images with two columns: directory and image name
        #masterf_frame: tuple with directory and frame name
        fd = cv_cl.featureDetectMatch()
        if not os.path.exists(directory_out):
            os.system('mkdir ' + directory_out)
        
        if not master_frame == None:
            master_img_name = master_frame
        else:   
            master_img_name = image_list[0]
        img_master = cv2.imread(master_img_name[0] + master_img_name[1])
        cv2.imwrite(os.path.join(directory_out, master_img_name[1])[:-4] + '_coreg.jpg', img_master)
              
        
        if master_0 == True:    #matchin to master
            '''detect Harris keypoints in master image'''
            keypoints_master, kp_draw = fd.HarrisCorners(master_img_name[0] + master_img_name[1], kp_nbr, False)
            if visualize:
                fd.plot_harris_points(img_master, kp_draw, True, directory_out)
            
            
            '''calculate ORB or SIFT descriptors in master image'''
            if not sift_vers:
                keypoints_master, descriptor_master = fd.OrbDescriptors(master_img_name[0] + master_img_name[1], 
                                                                        keypoints_master)
                print('ORB descriptors calculated for master ' + master_img_name[1])
            else: 
                keypoints_master, descriptor_master = fd.SiftDescriptors(master_img_name[0] + master_img_name[1], 
                                                                        keypoints_master)    
                print('SIFT descriptors calculated for master ' + master_img_name[1])
        
        
        '''border mask preparation (for temp texture)'''
        maskForBorderRegion_16UC1 = np.ones((img_master.shape[0], img_master.shape[1]))
        maskForBorderRegion_16UC1 = maskForBorderRegion_16UC1.astype(np.uint16)
        
        
        '''perform co-registration for each image'''
        i = 0
        while i < len(image_list):
                    
            if master_0 == False:   #matching always to subsequent frame (no master)
                '''skip first image (because usage of subsequent images)'''
                if i == 0:
                    i = i + 1
                    continue
                
                '''detect Harris keypoints in master image'''
                keypoints_master, kp_draw = fd.HarrisCorners(image_list[i-1][0] + image_list[i-1][1], kp_nbr, False)           
                
                '''calculate ORB or SIFT descriptors in master image'''
                if not sift_vers:
                    keypoints_master, descriptor_master = fd.OrbDescriptors(image_list[i-1][0] + image_list[i-1][1], 
                                                                            keypoints_master)
                    #print('ORB descriptors calculated for master ' + image_list[i-1][1])
                else: 
                    keypoints_master, descriptor_master = fd.SiftDescriptors(image_list[i-1][0] + image_list[i-1][1], 
                                                                            keypoints_master)    
                    #print('SIFT descriptors calculated for master ' + image_list[i-1][1])
            
             
            '''skip first image (because already read as master)'''
            if image_list[i][1] == master_img_name[1]:
                i = i + 1
                continue
        
        
            '''detect Harris keypoints in image to register'''
            keypoints_image, _ = fd.HarrisCorners(image_list[i][0] + image_list[i][1], kp_nbr, False)
        
            '''calculate ORB or SIFT descriptors in image to register'''
            if not sift_vers:
                keypoints_image, descriptor_image = fd.OrbDescriptors(image_list[i][0] + image_list[i][1], 
                                                                   keypoints_image)
                #print('ORB descriptors calculated for image ' + image_list[i][1])
            else:
                keypoints_image, descriptor_image = fd.SiftDescriptors(image_list[i][0] + image_list[i][1], 
                                                                   keypoints_image)
                #print('SIFT descriptors calculated for image ' + image_list[i][1])
            
            
            '''match images to master using feature descriptors (SIFT)'''  
            if not sift_vers:
                matched_pts_master, matched_pts_img = fd.match_DescriptorsBF(descriptor_master, descriptor_image, keypoints_master, keypoints_image,
                                                                             True,feature_match_twosided)
                matched_pts_master = np.asarray(matched_pts_master, dtype=np.float32)
                matched_pts_img = np.asarray(matched_pts_img, dtype=np.float32)
            else:
                if feature_match_twosided:        
                    matched_pts_master, matched_pts_img = fd.match_twosidedSift(descriptor_master, descriptor_image, keypoints_master, keypoints_image, "FLANN")    
                else:
                    matchscores = fd.SiftMatchFLANN(descriptor_master, descriptor_image)
                    matched_pts_master = np.float32([keypoints_master[m[0].queryIdx].pt for m in matchscores]).reshape(-1,2)
                    matched_pts_img = np.float32([keypoints_image[m[0].trainIdx].pt for m in matchscores]).reshape(-1,2)
            
            print 'number of matches: ' + str(matched_pts_master.shape[0])
        
            if visualize:
                fd.plot_matches(master_img_name[0] + master_img_name[1], image_list[i][0] + image_list[i][1], 
                                matched_pts_master, matched_pts_img, 10, True, directory_out)
                #print('master image and image ' + image_list[i][1] + ' matched')
            
            
            '''calculate homography from matched image points and co-register images with estimated 3x3 transformation'''
            if matched_pts_master.shape[0] > nbr_good_matches:
                # Calculate Homography
                H_matrix, _ = cv2.findHomography(matched_pts_img, matched_pts_master, cv2.RANSAC, 3)
                
                # Warp source image to destination based on homography
                img_src = cv2.imread(image_list[i][0] + image_list[i][1])
                img_coregistered = cv2.warpPerspective(img_src, H_matrix, (img_master.shape[1],img_master.shape[0]))      #cv2.PerspectiveTransform() for points only
                
                #save co-registered image
                cv2.imwrite(os.path.join(directory_out, image_list[i][1])[:-4] + '_coreg.jpg', img_coregistered)
                
                
                '''Mask for border region'''
                currentMask = np.ones((img_master.shape[0], img_master.shape[1]))
                currentMask = currentMask.astype(np.uint16)
                currentMask = cv2.warpPerspective(currentMask, H_matrix, (img_master.shape[1],img_master.shape[0]))
                maskForBorderRegion_16UC1 = maskForBorderRegion_16UC1 * currentMask
                
            i = i + 1   
        
        if frames:
            write_file = open(directory_out + 'mask_border_frames.txt', 'wb')
        else:
            write_file = open(directory_out + 'mask_border.txt', 'wb')            
        writer = csv.writer(write_file, delimiter=",")
        writer.writerows(maskForBorderRegion_16UC1)
        write_file.close()


'''prepare image data for processing'''
class ImagePreProcess:
    
    def __init__(self):
        pass

    '''convert 2D world coordinates to pixel values to get georaster'''
    def world2Pixel(self, geoMatrix, xy):
    #estimation of raster position according to point coordinates (e.g. for raster clipping with shape)
        ulX = geoMatrix[0]  #origin X
        ulY = geoMatrix[3]  #origin Y
        xDist = geoMatrix[1]    #pixel width
        yDist = geoMatrix[5]    #pixel height
    
        xy_len = np.shape(xy)[0]
        row = np.rint((xy[:,0] - np.ones(xy_len) * ulX) / xDist)
        col = np.rint((xy[:,1] - np.ones(xy_len) * ulY) / (yDist))
            
        return np.array(list([row, col])).T   #integer


    ''' clip raster with given polygon to keep information only within clip '''
    def raster_clip(self, ras_to_clip, geotrans, polygon, visualize=False, flipped_rows=False, world2Pix=True,
                    return_rasClip=False):
    #polygon is list of X and Y coordinates
        #transform coordinates of polygon vertices to pixel coordinates
        if world2Pix:
            poly_coo = self.world2Pixel(geotrans, polygon)
        else:
            poly_coo = np.asarray(polygon, dtype=np.uint)
        
        #determine min and max for image extent setting
        x_min = np.nanmin(poly_coo[:,0])
        if x_min < 0:
            x_min = 0
        y_min = np.nanmin(poly_coo[:,1])
        if y_min < 0:
            y_min = 0
        x_max = np.nanmax(poly_coo[:,0])
        if x_max > ras_to_clip.shape[1]:
            x_max = ras_to_clip.shape[1]
        y_max = np.nanmax(poly_coo[:,1])
        if y_max > ras_to_clip.shape[0]:
            y_max = ras_to_clip.shape[0]
            
        if y_min > y_max or x_min > x_max:
            print('error with raster extent')
            return         
        
        else:
            #define image with corresponding size
            img = Image.new('1', (int(x_max - x_min), int(y_max - y_min)))
            
            #set minimal necessary image extent according to polygon size
            poly_coo_x = poly_coo[:,0] - np.ones(poly_coo.shape[0])*x_min
            poly_coo_y = poly_coo[:,1] - np.ones(poly_coo.shape[0])*y_min
            poly_coo_sm = np.array([poly_coo_x, poly_coo_y]).T
            
            #draw image with mask as 1 and outside as 0
            poly_coo_flat = [y for x in poly_coo_sm.tolist() for y in x]    
            ImageDraw.Draw(img).polygon(poly_coo_flat, fill=1)
            del poly_coo_x, poly_coo_y, poly_coo_sm, poly_coo_flat
            
            #convert image to array, consider that rows and columns are switched in image format
            mask_list = []
            for pixel in iter(img.getdata()):
                mask_list.append(pixel)
            mask = np.array(mask_list).reshape(int(y_max-y_min), int(x_max-x_min))
            del img, mask_list
        
            #add offset rows and columns to obtain original raster size
            if flipped_rows:
                add_rows_down = np.zeros((int(y_min), int(x_max-x_min)))
                add_rows_up = np.zeros((int(ras_to_clip.shape[0]-y_max), int(x_max-x_min)))
            else:
                add_rows_down = np.zeros((int(ras_to_clip.shape[0]-y_max), int(x_max-x_min)))
                add_rows_up = np.zeros((int(y_min), int(x_max-x_min)))
                
            add_cols_left = np.zeros((int(ras_to_clip.shape[0]), int(x_min)))
            add_cols_right = np.zeros((int(ras_to_clip.shape[0]), int(ras_to_clip.shape[1]-x_max)))
            mask_final = np.vstack((add_rows_up, mask))
            mask_final = np.vstack((mask_final, add_rows_down))
            mask_final = np.hstack((add_cols_left, mask_final))
            mask_final = np.hstack((mask_final, add_cols_right))
                    
            #extract values within clip (from poylgon)                                  
            mask_final[mask_final==0]=np.nan
            ras_clipped = mask_final * ras_to_clip.reshape(mask_final.shape[0], mask_final.shape[1])
            
            ras_clipped_to_extent = np.delete(ras_clipped, np.s_[mask.shape[0] + add_rows_up.shape[0] : ras_clipped.shape[0]], 0)
            ras_clipped_to_extent = np.delete(ras_clipped_to_extent, np.s_[0 : add_rows_up.shape[0]], 0)
            ras_clipped_to_extent = np.delete(ras_clipped_to_extent, np.s_[mask.shape[1] + add_cols_left.shape[1] : ras_clipped.shape[1]], 1)
            ras_clipped_to_extent = np.delete(ras_clipped_to_extent, np.s_[0 : add_cols_left.shape[1]], 1)
            del mask
            
            if visualize:
                #plt.imshow(ras_to_clip)
                plt.imshow(ras_clipped)
                plt.show()

            if not return_rasClip:       
                return ras_clipped_to_extent
            else:
                return ras_clipped_to_extent, ras_clipped, np.asarray([x_min, y_min])
        
        
    '''perform image cropping to same size'''
    def crop_img(self, image_list, template_dir, template_frame):
        
        '''image_list: images, which will be cropped
           template_frame: image, which functions as mask for clipping'''
        for image in image_list:
            src_frame = cv2.imread(template_dir + template_frame)
            width_frame, height_frame = src_frame.shape[:2]
            scr_image = cv2.imread(image[0] + '/' + image[1])
            width_image, height_image = scr_image.shape[:2]
        
            clip_width = int((width_image - width_frame) / 2)
            clip_height = int((height_image - height_frame) / 2)        
        
            crop_img = scr_image[clip_width : width_image - clip_width, clip_height : height_image - clip_height]
            cv2.imwrite(template_dir + image[1][:-4] + '_crop.jpg', crop_img)
            
            del scr_image, src_frame
        
        print(str(len(image_list)) + ' images cropped')
        
        
    '''mask areas to speed up water line detection'''
    def masking_areas(self, img_array, waterline, buff_dist, water_side='right'):
        if water_side == 'right':
            waterside = 'left'
            landside = 'right'
        if water_side == 'left':
            waterside = 'right'
            landside = 'left'
        
        '''mask area to analyze'''
        #create buffer around waterline
        waterline = with_dataframe(waterline[:,0], waterline[:,1], False, False)
        waterline_approx = LineString(waterline)
        waterline_approx_coords = waterline_approx.coords[:]
        
        #plot_pts(img_array, waterline)
        
        '''mask of water side'''
        #cut buffer into water and land area
        waterline_offset_water = waterline_approx.parallel_offset(buff_dist/2, waterside)
        if waterline_offset_water.geom_type == 'MultiLineString':
            a = np.array([[0,0]])
            for line in waterline_offset_water:
                line_arr = np.asarray(line)
                a = np.append(a,line_arr, axis=0)
            a = np.delete(a, 0, 0)
            waterline_offset_water = LineString(a)
                
        waterline_offset_water_coords = waterline_offset_water.coords[:]
        
        #check, which ends of both lines connect
        dist_line_ends_1 = np.sqrt(np.square(waterline_offset_water_coords[-1][0]-waterline_approx_coords[0][0])
                                   + np.square(waterline_offset_water_coords[-1][1]-waterline_approx_coords[0][1]))
        dist_line_ends_2 = np.sqrt(np.square(waterline_offset_water_coords[-1][0]-waterline_approx_coords[-1][0])
                                   + np.square(waterline_offset_water_coords[-1][1]-waterline_approx_coords[-1][1]))
        if dist_line_ends_1 > dist_line_ends_2: #reverse array if false ends of both lines connect
            waterline_offset_water_coords = waterline_offset_water_coords[::-1]
        
        #create mask
        mask_water_of_waterline_coords = np.vstack((waterline_offset_water_coords, waterline_approx_coords))
        
        clip_water, water, _ = self.raster_clip(img_array, [0,1,0,0,0,1], mask_water_of_waterline_coords, False, False, False, True)
        del clip_water
    
        
        '''mask of land side'''
        #cut buffer into water and land area
        waterline_offset_land = waterline_approx.parallel_offset(buff_dist/2, landside)    
        if waterline_offset_land.geom_type == 'MultiLineString':
            a = np.array([[0,0]])
            for line in waterline_offset_land:
                line_arr = np.asarray(line)
                a = np.append(a,line_arr, axis=0)
            a = np.delete(a, 0, 0)
            waterline_offset_land = LineString(a)
        
        waterline_offset_land_coords = waterline_offset_land.coords[:]
        
        #check, which ends of both lines connect
        dist_line_ends_1 = np.sqrt(np.square(waterline_offset_land_coords[-1][0]-waterline_approx_coords[0][0])
                                   + np.square(waterline_offset_land_coords[-1][1]-waterline_approx_coords[0][1]))
        dist_line_ends_2 = np.sqrt(np.square(waterline_offset_land_coords[-1][0]-waterline_approx_coords[-1][0])
                                   + np.square(waterline_offset_land_coords[-1][1]-waterline_approx_coords[-1][1]))
        if dist_line_ends_1 > dist_line_ends_2: #reverse array if false ends of both lines connect
            waterline_offset_land_coords = waterline_offset_land_coords[::-1]
        
        mask_land_of_waterline_coords = np.vstack((waterline_offset_land_coords, waterline_approx_coords))
        
        clip_land, land, _ = self.raster_clip(img_array, [0,1,0,0,0,1], mask_land_of_waterline_coords, False, False, False, True)
        del clip_land
            
            
        '''merge left and right water side'''
        merged_sides = LineString(waterline_offset_water_coords).union(LineString(waterline_offset_land_coords))
        merged_coords = merged_sides.convex_hull.exterior.coords[:]
        LeftBottom = [np.nanmin(np.asarray(merged_coords)[:,0]), np.nanmin(np.asarray(merged_coords)[:,1])]
        clip, _, extent_global = self.raster_clip(img_array, [0,1,0,0,0,1], 
                                                  merged_coords, False, False, False, True)
        del extent_global
        land = self.raster_clip(land, [0,1,0,0,0,1], 
                                merged_coords, False, False, False, False)
        water = self.raster_clip(water, [0,1,0,0,0,1], 
                                 merged_coords, False, False, False, False)
       
         
        return clip, land, water, LeftBottom, merged_coords
        

'''perform image processing to detect water line'''      
class ImageProcess:
    
    def __init__(self):
        pass
    
        
    '''performing histogram equalization'''                
    def histeq_16bit(self, im,nbr_bins=2**16):
        '''histogram equalization'''
        #get image histogram
        imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
        #imhist[0] = 0
        
        cdf = imhist.cumsum() #cumulative distribution function
        cdf ** .5
        cdf = (2**16-1) * cdf / cdf[-1] #normalize
        #cdf = cdf / (2**16.)  #normalize
        
        #use linear interpolation of cdf to find new pixel values
        im2 = np.interp(im.flatten(),bins[:-1],cdf)
    
        return np.array(im2, int).reshape(im.shape)
    
    '''cast 16bit image 1 channel to 8bit image with 3 channels (to avoid info loss)'''    
    def visualise_16Bit1channel_to_3channel8bit(self, img_CV_16UC1):
        # histogram equalization
        max16U1 = np.max(img_CV_16UC1)
        min16U1 = np.min(img_CV_16UC1)
    
        if(max != 0 and min != 0):  #if max and min not 0
            scale = np.float(255) / (max16U1 - min16U1)
            img_8UC1 = img_CV_16UC1.astype(np.uint8)
            img_8UC1 = cv2.convertScaleAbs(img_CV_16UC1, img_8UC1, scale, -1 * min16U1 * scale)
    
            # use ColorMap
            falseColorMap_8UC3_Mat = cv2.applyColorMap(img_8UC1, cv2.COLORMAP_RAINBOW)
            
            return falseColorMap_8UC3_Mat
            
    '''Map a 16-bit image trough a lookup table to convert it to 8-bit'''
    def map_uint16_to_uint8(self, img, lower_bound=None, upper_bound=None):
        
        if not(0 <= lower_bound < 2**16) and lower_bound is not None:
            raise ValueError(
        '"lower_bound" must be in the range [0, 65535]')
        if not(0 <= upper_bound < 2**16) and upper_bound is not None:
            raise ValueError(
        '"upper_bound" must be in the range [0, 65535]')
        if lower_bound is None:
            lower_bound = np.min(img)
        if upper_bound is None:
            upper_bound = np.max(img)
        if lower_bound >= upper_bound:
            raise ValueError(
        '"lower_bound" must be smaller than "upper_bound"')
        lut = np.concatenate([
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
        ])
        return lut[img].astype(np.uint8)

    
    '''calculate temporal texture and average image'''
    # script adapted from Melanie Kroehnert
    def tempTexture(self, image_list, directory_out, border_mask_file=None, bilat_filter_img=False, bilat_filter_tempText=False):
        
        if not os.path.exists(directory_out):
            os.system('mkdir ' + directory_out)
        
        master_img_name = image_list[0]
        img_master_8U3C = cv2.imread(master_img_name[0] + master_img_name[1], cv2.IMREAD_COLOR)  #image read as 8bit color (3 channel image)
        img_master_16U3C = img_master_8U3C.astype(np.uint16)     #image converted to 16bit color (for average image)
        img_master_16U1C = cv2.cvtColor(img_master_16U3C, cv2.COLOR_BGR2GRAY)   #image converted to 16bit gray scale (for temporal texture)
        print('master image ' + master_img_name[1] + ' prepared')
        
        '''read border mask of co-registered images'''
        if not border_mask_file==None:
            try:
                mask_border_read = open(border_mask_file, 'rb')
                mask_border = csv.reader(mask_border_read, delimiter=',')
                mask_border_16UC1 = []
                for line in mask_border:
                    mask_border_16UC1.append(line)
                mask_border_16UC1 = np.asarray(mask_border_16UC1, dtype=np.uint16)
            except:
                print('border mask file not readable')
                return None, None
        
        
        '''Calculation average image and temporal texture'''
        img_average_16U3C = img_master_16U3C
        img_tempTexture_16UC1 = np.zeros((img_master_16U3C.shape[0], img_master_16U3C.shape[1]))
        count_avg = 0
        for image_name in image_list:
            
            '''skip first image (because already read as master)'''
            if image_name[1] == master_img_name[1]:
                continue
            
            '''average img preparation'''
            img_8U3C = cv2.imread(image_name[0] + image_name[1], cv2.IMREAD_COLOR)   #image read as 8bit color (3 channel image)        
    
            if bilat_filter_img:
                img_8U3C = cv2.bilateralFilter(img_8U3C, 3, 300, 300)
                        
            img_16U3C = img_8U3C.astype(np.uint16)  #image converted to 16bit color (for average image)
            img_average_16U3C = img_average_16U3C + img_16U3C   #image converted to 16bit gray scale (for temporal texture)
            
            '''temp texture preparation'''
            img_16U1C = cv2.cvtColor(img_16U3C, cv2.COLOR_BGR2GRAY)
            img_temp_diff_16UC1 = np.sqrt(np.square((img_master_16U1C - img_16U1C))).astype(np.uint16)
            img_tempTexture_16UC1 = img_tempTexture_16UC1 + img_temp_diff_16UC1   
        
            count_avg = count_avg + 1
        
        
        '''average image'''
        img_average_16U3C = img_average_16U3C / count_avg
        #map 16bit to 8bit image
        img_average_8U3C = self.map_uint16_to_uint8(img_average_16U3C)
        img_average_8U1C = cv2.cvtColor(img_average_8U3C, cv2.COLOR_BGR2GRAY)        
        
        '''correct for border from co-registration'''
        #print np.nanmin(mask_border_16UC1), np.nanmax(mask_border_16UC1)
        if not border_mask_file==None:
            mask_border_16UC3 = mask_border_16UC1
            for channel in range(2):
                mask_border_16UC3 = np.dstack((mask_border_16UC3, mask_border_16UC1))
            del channel
            img_average_8U3C = img_average_8U3C * mask_border_16UC3.astype(np.uint8)
        else:
            img_average_8U3C = img_average_8U3C.astype(np.uint8)
        
        cv2.imwrite(os.path.join(directory_out, master_img_name[1])[:-4] + '_average.jpg', img_average_8U3C)
        print('average image saved')
        
        
        '''temporal texture'''   
        if not border_mask_file==None:
            img_tempTexture_16UC1 = (img_tempTexture_16UC1 * mask_border_16UC1)
            img_tempTexture_16UC1 = img_tempTexture_16UC1.astype(np.uint16)       
        else:
            img_tempTexture_16UC1 = img_tempTexture_16UC1.astype(np.uint16)
        
        
        '''work with 16bit'''
        #perform histogram normalization to increase contrast between change and no change
        #img_tempTexture_16UC1_Norm = cv2.normalize(img_tempTexture_16UC1, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        
        #cast 16bit 1 channel information to 3 channel 8bit image for illustration    
        img_tempTexture_16UC1_Equal_print = self.histeq_16bit(img_tempTexture_16UC1).astype(np.uint16)  
        img_tempTexture_16UC1_Equal_8U3C = self.visualise_16Bit1channel_to_3channel8bit(img_tempTexture_16UC1_Equal_print)    
        cv2.imwrite(os.path.join(directory_out, master_img_name[1])[:-4] + '_tempTexture_16bit.jpg', img_tempTexture_16UC1_Equal_8U3C)
            
        
        '''work with 8bit'''
        #map 16bit to 8bit image
        img_tempTexture_8U1C = self.map_uint16_to_uint8(img_tempTexture_16UC1)
        
        #perform histogram equalization to increase contrast between change and no change    
        img_tempTexture_8UC1_HistEqual_print = cv2.equalizeHist(img_tempTexture_8U1C)
        cv2.imwrite(os.path.join(directory_out, master_img_name[1])[:-4] + '_tempTexture_8bit.jpg', img_tempTexture_8UC1_HistEqual_print)  
        
        #bilateral filtering
        if bilat_filter_tempText:
            img_tempTexture_8UC1_filt = cv2.bilateralFilter(img_tempTexture_8U1C, 13, 75, 75)    #
            img_tempTexture_8UC1_filt = cv2.medianBlur(img_tempTexture_8UC1_filt, 7)  #
        
            #save img
            img_tempTexture_8UC1_HistEqual_print = cv2.equalizeHist(img_tempTexture_8UC1_filt)
            img_tempTexture_8UC1_HistEqual_print = cv2.equalizeHist(img_tempTexture_8UC1_HistEqual_print)
            cv2.imwrite(os.path.join(directory_out, master_img_name[1])[:-4] + '_tempTexture_8bit_filtered.jpg', img_tempTexture_8UC1_HistEqual_print)
        
        
        print('temporal texture saved')
        
        return img_average_8U1C, img_tempTexture_16UC1 


'''calculate histograms for comparison'''
class Histogram:
    
    def __init__(self):
        pass
    
    
    '''perform histogram correlation and find maximum of selected histogram'''
    def histogram_analysis_waterline(self, clip_all, clip_land, clip_water, directory_output, plot_results=False):
        draw_tools = drw.Drawing()
        
        min_val_hist = 4
        
        #land side
        bin_number = np.nanmax(clip_all)
        tempText_land_no0 = clip_land[clip_land[:,:] > min_val_hist]
        hist_land = cv2.calcHist([tempText_land_no0.astype(np.float32)], [0] , None, [bin_number] ,[0, bin_number])
        
        hist_land_vals = cv2.minMaxLoc(hist_land)
        hist_land_maxX, hist_land_maxY = hist_land_vals[3]   
        del hist_land_maxX
        print('maximum histogram value (land side): ' + str(hist_land_maxY))
    
        if plot_results:
            plt.clf()
            plt.plot(hist_land)
            plt.title('Histogram land')
            plt.savefig(os.path.join(directory_output, 'histogram_land.jpg'),  dpi=600) 
            plt.close()
        
        
        #water side 
        tempText_water_no0 = clip_water[clip_water[:,:] > min_val_hist]    #to consider too bright images
        hist_water = cv2.calcHist([tempText_water_no0.astype(np.float32)], [0] , None, [bin_number] ,[0, bin_number])
        
        if plot_results:  
            plt.clf()
            plt.plot(hist_water)
            plt.title('Histogram water')
            plt.savefig(os.path.join(directory_output, 'histogram_water.jpg'),  dpi=600) 
            plt.close()
            
        corr_hist = cv2.compareHist(hist_land, hist_water, cv2.cv.CV_COMP_CORREL)
        
        
        #check if more than one peak in histogram
        if corr_hist >= 0.95:
         
            hist_water_smooth = movingaverage(hist_water, 75)
            i = 2
            peak = []
            while i+2 < len(hist_water_smooth):
                if(hist_water_smooth[i]>hist_water_smooth[i-1] and hist_water_smooth[i]>hist_water_smooth[i+1]
                   and hist_water_smooth[i]>hist_water_smooth[i-2] and hist_water_smooth[i]>hist_water_smooth[i+2]):
                    if hist_water_smooth[i] > 10:
                        peak.append(i)
                i = i + 1
            #reset correlation because water histogram corrupted by land
            if len(peak) > 1:
                corr_hist = 0
                print(peak)
                
            if plot_results:  
                plt.clf()
                plt.plot(hist_water_smooth)
                plt.title('Histogram smooth water')
                plt.savefig(os.path.join(directory_output, 'histogram_smooth_water.jpg'),  dpi=600) 
                plt.close()
                
        print('Correlation coefficient histogram comparison: ' + str(corr_hist))
        
        return corr_hist, hist_land_maxY + min_val_hist


'''perform region growing'''
class RegionGrow:
    
    def __init__(self):
        pass
    
    
    '''define seed points for region growing'''
    def define_seedpts(self, clip_steady, hist_steady_thresh):
        seed_points = []
        seed_x = 0
        seed_y = 0
        #define seed points
        for seed_point_row in clip_steady:    #tempText_clip_f_16bit
            for seed_point in seed_point_row:
                if seed_point == hist_steady_thresh: #> hist_land_maxY-200 and seed_point < hist_land_maxY+200
                    seed_points.append([seed_y, seed_x])
                seed_x = seed_x + 1
            seed_y = seed_y + 1
            seed_x = 0
        
        return seed_points
    
    
    '''perform region growing'''
    def simple_region_growing(self, img, seed, thresholdHist=1, reg=np.empty((1,1))):
    
        seed = tuple(seed)
        
        try:
            dims = img.shape
        except TypeError:
            raise TypeError("(%s) img : IplImage expected!" % (sys._getframe().f_code.co_name))
    
        # threshold tests
        if (not isinstance(thresholdHist, int)) :
            raise TypeError("(%s) Int expected!" % (sys._getframe().f_code.co_name))
        elif thresholdHist < 0:
            raise ValueError("(%s) Positive value expected!" % (sys._getframe().f_code.co_name))
        # seed tests
        if not((isinstance(seed, tuple)) and (len(seed) is 2) ) :
            raise TypeError("(%s) (x, y) variable expected!" % (sys._getframe().f_code.co_name))
    
        if (seed[0] or seed[1] ) < 0 :
            raise ValueError("(%s) Seed should have positive values!" % (sys._getframe().f_code.co_name))
        elif ((seed[0] > dims[0]) or (seed[1] > dims[1])):
            raise ValueError("(%s) Seed values greater than img size!" % (sys._getframe().f_code.co_name))
    
        if reg.shape[0] == 1:
            reg = np.zeros((dims[0], dims[1]))
    
        orient = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
    #     orient_sides = [[(1,0), (1, -1), (1,1), (1,-2), (1,2)], [(0,1), (-1, 1), (1,1), (-2,1), (2,1)], 
    #                     [(-1,0), (-1, -1), (-1,1), (-1,-2), (-1,2)], [(0,-1), (-1, -1), (1,-1), (-2,-1), (2,-1)]]
    #     orient = [(1, 0), (0, 1), (-1, 0), (0, -1), (1,-1), (1,1), (-1,1), (-1,-1)] # 8 connectivity
        cur_pix = [seed[0], seed[1]]
    
        regionPtsList = []
        regionPtsList.append(cur_pix)
    
        #Spreading
        for cur_pix in regionPtsList:
        #adding pixels
            for j in range(len(orient)):
                #select new candidate
                temp_pix = [cur_pix[0] + orient[j][0], cur_pix[1] + orient[j][1]]                
                    
                #check if it belongs to the image
                is_in_img = dims[0]>temp_pix[0]>0 and dims[1]>temp_pix[1]>0 #returns boolean
                 
                #candidate is taken if not already selected before
                if (is_in_img and (reg[temp_pix[0], temp_pix[1]]==0)) and (img[temp_pix[0], temp_pix[1]] <= thresholdHist):
                    #append point to region
                    regionPtsList.append(temp_pix)
                    reg[temp_pix[0], temp_pix[1]] = 255          
    
            reg[cur_pix[0], cur_pix[1]] = 255
                
        return reg
   

'''detect water line'''
class WaterlineEstimation:
    
    def __init__(self):
        pass
 
           
    def waterline_estimate(self, temporalTexture, averageImg, directory_output, waterline_approx_name, plot_results=False, 
                           thresh_correlation_histogram=0.5, buff_dist=50, waterside='right', thresh_add_tempText=0, thresh_add_grey=0,
                           kernel_bilat = [7,3], canny_kernel = 5, use_canny=True, nan_clip_size = None):
        #kernel_bilat = [13, 7]  defintion of kernel sizes for filtering -> [mean_bilat, median]); Elbersdorf: [17?13, 9?7], Trieb: [11, 5]
        draw_tools = drw.Drawing()
        imPrePro = ImagePreProcess()
        hist = Histogram()
        ptLine = LineAndPointOperation()
        rg = RegionGrow()
        
        try:
            clust_dist = 5
            if nan_clip_size == None:
                nan_clip_size = 5  #Triebe: 40,50, Elbers: 40
        
            if waterside == 'right':
                landside = 'left'
            else:
                landside = 'right'
                        
            skip_region_grow = True
            
            #prepare logfile
            log_file = open(os.path.join(directory_output, 'logfile.txt'), 'wb')
            writer = csv.writer(log_file, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            
            '''preparation to calculate mask around waterline'''
            #get approximated waterline
            waterline_approx_file = open(waterline_approx_name, 'rb')
            waterline_approx_read = csv.reader(waterline_approx_file, delimiter=',')
            waterline_approx_table = []
            for line in waterline_approx_read:
                waterline_approx_table.append(line)
            waterline_approx_table = np.asarray(waterline_approx_table, dtype=np.float64)
            waterline_approx_table = waterline_approx_table.astype(np.uint)
            waterline_approx_table_sw = waterline_approx_table.astype(np.uint)
            waterline_approx_table_sw[:,0] =  waterline_approx_table[:,1]
            waterline_approx_table_sw[:,1] =  waterline_approx_table[:,0]
            waterline_approx_file.close()  
           
                   
            '''calculate mask'''
            tempText_clip, tempText_land, tempText_water, LeftBottom, convexHull = imPrePro.masking_areas(temporalTexture, waterline_approx_table, 
                                                                                                          buff_dist, waterside)
            
            if plot_results:
                plt.clf()
                plt.imshow(tempText_land)
                plt.title('land side')
                plt.savefig(os.path.join(directory_output, 'clip_land.jpg'),  dpi=600) 
                plt.close()
                plt.clf()
                plt.imshow(tempText_water)
                plt.title('water side')
                plt.savefig(os.path.join(directory_output, 'clip_water.jpg'),  dpi=600) 
                plt.close()    
            
            
            '''histogram analysis'''
            corr_hist, hist_land_maxY = hist.histogram_analysis_waterline(tempText_clip, tempText_land, 
                                                                          tempText_water, directory_output, plot_results)
            hist_land_maxY = hist_land_maxY + thresh_add_tempText    #add some "buffer" to ensure all land pixels included during region grow
            writer.writerow(['Correlation'] + [str(corr_hist)])
            log_file.flush()
                
            
            '''preparation erosion of clipped area to delete bounding box as contour'''
            tempText_clip_nan = np.isnan(tempText_clip)
            tempText_clip_nan = skimage.morphology.dilation(tempText_clip_nan, square(nan_clip_size))  
            tempText_clip_nan[-nan_clip_size:,:] = True
            
            
            '''if water surface enough changes start region growing from land side'''
            #for clustering define waterline_approx in middle of clip
            slope_water_approx = ((np.float(waterline_approx_table[1,1]) - np.float(waterline_approx_table[0,1])) / 
                                  (np.float(waterline_approx_table[1,0]) - np.float(waterline_approx_table[0,0])))
            waterl_approx_len = np.sqrt(np.square(waterline_approx_table[1,1] - waterline_approx_table[0,1]) + 
                                        np.square(waterline_approx_table[1,0] - waterline_approx_table[0,0]))
            water_approx_clip_pos = [tempText_clip.shape[1]/2 + waterl_approx_len/2, tempText_clip.shape[0]/2 + (waterl_approx_len/2 * slope_water_approx)]
            water_approx_clip_neg = [tempText_clip.shape[1]/2 - waterl_approx_len/2, tempText_clip.shape[0]/2 - (waterl_approx_len/2 * slope_water_approx)]
            waterline_clip = np.vstack((water_approx_clip_pos, water_approx_clip_neg))
            
            #decide if waterline detection from land or water side
            if use_canny:    
                perform_canny = True
                perform_region_greyscale = False
            else:
                perform_canny = False
                perform_region_greyscale = True
                
            if corr_hist < thresh_correlation_histogram:      #if temporal texture signficant than perform region grow   
                perform_canny = False   #no waterline detection from water side (with canny) needed
                perform_region_greyscale = False
                
                
                '''filter clipped area'''   
                #16bit (not already during temporal texture calculation because takes too long)
                tempText_clip_f_16bit = mean_bilateral(tempText_clip.astype(np.uint16), disk(kernel_bilat[0])).astype(np.uint16) 
                tempText_clip_f_16bit = median(tempText_clip_f_16bit, disk(kernel_bilat[1])).astype(np.uint16)    
            
                if plot_results:
                    plt.clf()
                    plt.imshow(tempText_clip_f_16bit)
                    plt.title('filtered temporal texture 16bit')
                    plt.savefig(os.path.join(directory_output, 'clip_tempText.jpg'),  dpi=600) 
                    plt.close()
            
                ''''find seed points'''
                seed_points = rg.define_seedpts(tempText_land, hist_land_maxY)    
                    
                if not len(seed_points) > 0:
                    print('No seed points for region growing detected')
                    if use_canny:
                        perform_canny = True
                    else:
                        perform_region_greyscale = True
                    corr_hist = 1
                    skip_region_grow = True
                
                else:
                    print('Number of seed points for region growing: ' + str(len(seed_points)))
                    skip_region_grow = False
                    if plot_results:
                        draw_tools.plot_pts(tempText_clip_f_16bit, seed_points, True, 'Seed points', True)
                        plt.title('seed points')
                        plt.savefig(os.path.join(directory_output, 'seed_points.jpg'),  dpi=600) 
                        plt.close()
    
            
            if not skip_region_grow:
                #perform region growing for all seed points
                try:
                    '''setup mask for region growing to avoid growing in no-area'''
                    tempText_clip_nan_mask = np.ones((tempText_clip.shape[0], tempText_clip.shape[1]))
                    tempText_clip_nan_bool = np.isnan(tempText_clip)
                    tempText_clip_nan_mask[tempText_clip_nan_bool] = np.nan
                    
                    '''start region growing'''
                    tempText_clip_f_16bit_nan = tempText_clip_nan_mask * tempText_clip_f_16bit
                    regionGrown = rg.simple_region_growing(tempText_clip_f_16bit_nan, seed_points[0], hist_land_maxY-int(thresh_add_tempText/2))  #hist_land_maxY + 5
                    for seed_point in seed_points:
                        regionGrown = rg.simple_region_growing(tempText_clip_f_16bit_nan, seed_point, hist_land_maxY-int(thresh_add_tempText/2), regionGrown) #hist_land_maxY + 5
                    regionGrown = regionGrown.astype(np.uint8)
        
        
                    if plot_results:
                        plt.clf()
                        plt.imshow(regionGrown)
                        plt.title('region grow result')
                        plt.savefig(os.path.join(directory_output, 'regionGrow_area.jpg'),  dpi=600) 
                        plt.close()                
                
                    '''detect contours of regions and select largest boundary'''
                    waterline = ptLine.detect_contours_RG(regionGrown, tempText_clip_nan)
                    
                    if plot_results:
                        draw_tools.plot_pts(tempText_clip_f_16bit, waterline, False, 'final contour', True) 
                        plt.savefig(os.path.join(directory_output, 'regionGrow_contour.jpg'),  dpi=600)
                        plt.close()        
                        
                    
                    if waterline.shape[0] <= 1:  #if waterline detection failed try again from water side with canny
                        perform_canny = True
                        corr_hist = 1
                    else:
                        '''cluster points if several water lines retrieved due to masking (erosion)'''
                        plot_clust, waterline = ptLine.analyse_for_clusters(waterline, cluster.DBSCAN, (), {'eps':clust_dist}, waterline_clip, landside, 
                                                                            tempText_clip_f_16bit.shape[0], plot_results)
                        
                        if plot_results and plot_clust != None:
                            plot_clust.savefig(os.path.join(directory_output, 'clustered_RG_results.jpg'),  dpi=600) 
                            plot_clust.close()
                    
                    print('performed from land side')
                    writer.writerow(['Water line detection performed from land side (region grow)'])
                    log_file.flush()            
        
                except Exception as e:               
                    print('region grow temp texture failed')
                    if use_canny:
                        perform_canny = True
                    else:
                        perform_region_greyscale = True
                        
                    skip_region_grow = True
                    
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno) 
                    
        
            '''if changes at water surface too small use water surface to detect waterline'''
            if perform_canny:
                avgImg_clip_8U1C, avgImg_land_8U1C, avgImg_water_8U1C, _, __ = imPrePro.masking_areas(averageImg, waterline_approx_table, 
                                                                                                      buff_dist, waterside)
                del avgImg_land_8U1C, avgImg_water_8U1C
                
                #bilateral filtering
                avgImg_clip_blur_8U1C = cv2.GaussianBlur(avgImg_clip_8U1C.astype(np.uint8), (canny_kernel, canny_kernel), 0)
                avgImg_clip_bilat_8U1C = cv2.bilateralFilter(avgImg_clip_blur_8U1C.astype(np.uint8), canny_kernel*3+1, 75, 75)    #Elbersdorf: 31, 75, 75; Trieb: 13, 75, 75
        
                if plot_results:
                    plt.clf()
                    plt.imshow(avgImg_clip_bilat_8U1C)
                    plt.title('bilateral filter')
                    plt.savefig(os.path.join(directory_output, 'bilaterat_filter.jpg'),  dpi=600) 
                    plt.close()    
                    
                
                #Canny edge detection with otsu
                avgImg_clip_blur_8U1C = cv2.GaussianBlur(avgImg_clip_bilat_8U1C.astype(np.uint8), (canny_kernel, canny_kernel), 0)    #0.3*((kernel-1)*0.5 - 1) + 0.8
                otsu_thresh_val, _ = cv2.threshold(avgImg_clip_bilat_8U1C, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                avgImg_canny_binary = cv2.Canny(avgImg_clip_blur_8U1C, otsu_thresh_val, otsu_thresh_val * 0.5)        
                
                if plot_results:    
                    plt.clf()
                    plt.imshow(avgImg_canny_binary)
                    plt.title('canny filtered image')
                    plt.savefig(os.path.join(directory_output, 'clip_canny.jpg'),  dpi=600) 
                    plt.close()
        
                
                waterline_approx_len = LineString(waterline_approx_table).length
                waterline = ptLine.detect_contours_Canny(avgImg_canny_binary, waterline_approx_len, tempText_clip_nan)
                
                '''cluster points if several water lines retrieved due to masking (erosion)'''        
                plot_clust, waterline = ptLine.analyse_for_clusters(waterline, cluster.DBSCAN, (), {'eps':clust_dist}, waterline_clip, landside, 
                                                             avgImg_clip_8U1C.shape[0], plot_results, True)
                
                if plot_results and plot_clust != None:
                    plot_clust.savefig(os.path.join(directory_output, 'clustered_RG_results.jpg'),  dpi=600) 
                    plot_clust.close()
        
        
                print('performed from water side')
                writer.writerow(['Water line detection performed from water side (canny)'])
                log_file.flush()                
    
        
            ''''if river shore too complex better to use water detection via "darkness"'''
            if perform_region_greyscale:
                min_val_hist = 4
                
                '''equqlize histogram'''
                averageImg_equ = cv2.equalizeHist(averageImg)
                
                '''calculate mask'''
                Gray_clip, Gray_land, Gray_water, LeftBottom, convexHull = imPrePro.masking_areas(averageImg_equ, waterline_approx_table, 
                                                                                                  buff_dist, waterside)   
                
                '''setup mask for region growing to avoid growing in no-area'''
                Gray_clip_nan_mask = np.ones((Gray_clip.shape[0], Gray_clip.shape[1]))
                Gray_clip_nan_bool = np.isnan(Gray_clip)
                Gray_clip_nan_mask[Gray_clip_nan_bool] = np.nan
                
                        
                '''filter clipped area'''   
                #16bit (not already during temporal texture calculation because takes too long)
                gray_clip_filt = mean_bilateral(Gray_clip.astype(np.uint8), disk(kernel_bilat[0]-4)).astype(np.uint8) 
                gray_clip_filt = median(gray_clip_filt, disk(kernel_bilat[1]+2)).astype(np.uint8)  
                
                ''''histogram of water side'''
                bin_number = np.nanmax(Gray_clip)
                gray_water_no0 = Gray_water[Gray_water[:,:] > min_val_hist]
                hist_water = cv2.calcHist([gray_water_no0.astype(np.float32)], [0] , None, [bin_number] ,[0, bin_number])
                
                hist_water_vals = cv2.minMaxLoc(hist_water)
                hist_water_maxX, hist_water_maxY = hist_water_vals[3]
                del hist_water_maxX
                
                if plot_results:
                    plt.clf()
                    plt.plot(hist_water)
                    plt.title('Histogram water grey')
                    plt.savefig(os.path.join(directory_output, 'histogram_water_grey.jpg'),  dpi=600) 
                    plt.close()
                print('maximum histogram value (grey): ' + str(hist_water_maxY))
                
                
                '''find seed points'''
                seed_points = rg.define_seedpts(Gray_water, hist_water_maxY)   #-5
                
                print('Number of seed points for region growing: ' + str(len(seed_points)))
                if plot_results:
                    draw_tools.plot_pts(gray_clip_filt, seed_points, True, 'Seed points', True)
                    plt.title('seed points')
                    plt.savefig(os.path.join(directory_output, 'seed_points_grey.jpg'),  dpi=600) 
                    plt.close()
        
        
                '''perform region growing for all seed points'''
                gray_clip_filt_nan = Gray_clip_nan_mask * gray_clip_filt
                regionGrown_GS = rg.simple_region_growing(gray_clip_filt_nan, seed_points[0], hist_water_maxY + thresh_add_grey)  #hist_land_maxY + 5
                for seed_point in seed_points:
                    regionGrown_GS = rg.simple_region_growing(gray_clip_filt_nan, seed_point, hist_water_maxY + thresh_add_grey, regionGrown_GS) #hist_land_maxY + 5
                regionGrown_GS = regionGrown_GS.astype(np.uint8)
        
                if plot_results:
                    plt.clf()
                    plt.imshow(regionGrown_GS)
                    plt.title('region grow result (Grayscale)')
                    plt.savefig(os.path.join(directory_output, 'regionGrow_area_grayscale.jpg'),  dpi=600) 
                    plt.close()   
        
                    
                '''detect contours of regions and select largest boundary'''
                waterline = ptLine.detect_contours_RG(regionGrown_GS, tempText_clip_nan)                        
                if plot_results:
                    plt.clf()
                    draw_tools.plot_pts(regionGrown_GS, waterline, False, 'Largest contour', True)
                    plt.savefig(os.path.join(directory_output, 'largest_contours_RGgrey.jpg'),  dpi=600) 
                    plt.close()   
                                
                
                '''cluster points if several water lines retrieved due to masking'''
                plot_clust, waterline = ptLine.analyse_for_clusters(waterline, cluster.DBSCAN, (), {'eps':clust_dist}, waterline_clip, waterside, 
                                                             gray_clip_filt.shape[0], plot_results, True)
                
                if plot_results and plot_clust != None:
                    plot_clust.savefig(os.path.join(directory_output, 'clustered_RGgrey_results.jpg'),  dpi=600) 
                    plot_clust.close()
    
        
            '''use non-linear regression to clean waterline from outliers'''
            waterline_smooth = ptLine.rotate_smooth_rotateback(waterline)
            
            #print(waterline_smooth)
        
            writer.writerow(['Water line length'] + [str(LineString(waterline_smooth).length)])
            log_file.flush()
            log_file.close()  
                        
            waterline_smooth = np.asarray(waterline_smooth)
            addValX = np.ones((waterline_smooth.shape[0], 1)) * LeftBottom[1]
            addValY = np.ones((waterline_smooth.shape[0], 1)) * LeftBottom[0]
            addVal = np.hstack((addValX, addValY))
            waterline_smooth = waterline_smooth + addVal
            waterline_final = with_dataframe(waterline_smooth[:,0], waterline_smooth[:,1], True, True)
            waterline_final_txt = np.empty((waterline_final.shape))
            waterline_final_txt[:,0] = waterline_final[:,1]
            waterline_final_txt[:,1] = waterline_final[:,0]
            
            '''output waterline'''
            write_file = open(directory_output + '/wasserlinie.txt', 'wb')
            writer = csv.writer(write_file, delimiter=",")
            writer.writerows(waterline_final_txt)
            write_file.close()
            
            #if plot_results:
            draw_tools.draw_points_onto_image(averageImg, waterline_final, [], 6, 10, True) 
            plt.savefig(os.path.join(directory_output, 'waterline.jpg'),  dpi=600)
            plt.close('all')
        
            return waterline_final_txt, convexHull, corr_hist
        
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)   
