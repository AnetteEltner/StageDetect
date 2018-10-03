import os, sys
from PIL import Image
import numpy as np
import pylab as plt
import cv2

import draw_classes as drw


#detect and match image features using algorithms from computer vision
class featureDetectMatch:
    
    def __init__(self):
        pass
    
    #detect SIFT features using patented SIFT library from Lowe
    def process_image(self, imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
        '''Process an image and save the results in a file'''
        
        if imagename[-3:] != 'pgm':
            #create a pgm file
            im = Image.open(imagename).convert('L')
            im.save('tmp.pgm')
            imagename = 'tmp.pgm'
            
        cmmd = str("sift " + imagename + " --output=" + resultname + " " + params)
        os.system(cmmd)
        print 'processed', imagename, 'to', resultname
     
    #read detected SIFT features (including feature description) from saved file
    def read_features_from_file(self, filename):
        '''Read feature properties and return in matrix form.'''
        
        f = np.loadtxt(filename)
        return f[:,:4], f[:,4:] #feature locations, descriptors
    
    #save detected features including descriptor to file
    def write_features_to_file(self, filename, locs, desc):
        '''Save feature location and descriptor to file.'''
        np.savetxt(filename, np.hstack((locs, desc)))
    
    #detect features using STAR detector
    def find_describe_STAR(self, imagename):
        '''Find and compute descriptors of STAR features'''
        img = cv2.imread(imagename)
            
        orb = cv2.ORB()
        kp = orb.detect(img,None)
        
        kp, des = orb.compute(img, kp)
        
        print('STAR keypoint descriptors calculated: ' + str(len(kp)))
        
        return kp, des
    
    #match STAR image features using bruce force matching
    def match_DescriptorsBF(self, des1,des2,kp1,kp2,ratio_test=True,twosided=True):
        '''Match STAR descriptors between two images'''
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(des1,des2)                    
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)     
        
        pts1 = []
        pts2 = []        
        
        if ratio_test: 
            # ratio test as per Lowe's paper
            good = []
            for m in matches:
                if m.distance < 100:
                    good.append(m)
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)
        else:
            for m in matches:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                
        if twosided:
            pts1_b = []
            pts2_b = []
            
            matches_back = bf.match(des2,des1)
            for m in matches_back:
                pts2_b.append(kp1[m.trainIdx].pt)
                pts1_b.append(kp2[m.queryIdx].pt)
            
            pts1_arr = np.asarray(pts1)
            pts2_arr = np.asarray(pts2)
            pts_12 = np.hstack((pts1_arr, pts2_arr))
            pts1_arr_b = np.asarray(pts1_b)
            pts2_arr_b = np.asarray(pts2_b)        
            pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))
           
            
            pts1_ts = []
            pts2_ts = []        
            for pts in pts_12:
                pts_comp = np.asarray(pts, dtype = np.int)
                for pts_b in pts_21:
                    pts_b_comp = np.asarray(pts_b, dtype = np.int)
                    if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                        and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                        pts1_ts.append(pts[0:2].tolist())
                        pts2_ts.append(pts[2:4].tolist())
                        
                        break
            
            pts1 = pts1_ts
            pts2 = pts2_ts      
            
            #print('Matches calculated')
                
        return pts1, pts2
    
    #match SIFT image features using bruce force matching    
    def SiftMatchBF(self, des1, des2):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        return good
    
    #match SIFT image features using FLANN matching
    def SiftMatchFLANN(self, des1,des2):
        max_dist = 0
        min_dist = 100
        
        # FLANN parameters   
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)
        
        matches = flann.knnMatch(des1,des2,k=2)
           
        # ratio test as per Lowe's paper
        for m,n in matches:
            if min_dist > n.distance:
                min_dist = n.distance
            if max_dist < n.distance:
                max_dist = n.distance
        
        good = []
        for m,n in matches:
            #if m.distance < 0.75*n.distance:
            if m.distance <= 3*min_dist:
                good.append([m])
        
        return good
    
    #match SIFT image features using FLANN matching and perform two-sided matching
    def match_twosidedSift(self, desc1, desc2, kp1, kp2, match_Variant="FLANN"):
        '''Two-sided symmetric version of match().'''
        
        if match_Variant == "FLANN":
            matches_12 = self.SiftMatchFLANN(desc1,desc2)
            matches_21 = self.SiftMatchFLANN(desc2,desc1)
        elif match_Variant == "BF":
            matches_12 = self.SiftMatchBF(desc1,desc2)
            matches_21 = self.SiftMatchBF(desc2,desc1)
    
        pts1 = []
        pts2 = []
        for m in matches_12:
            pts1.append(kp1[m[0].queryIdx].pt)
            pts2.append(kp2[m[0].trainIdx].pt)
    
        pts1_b = []
        pts2_b = []    
        for m in matches_21:
            pts2_b.append(kp1[m[0].trainIdx].pt)
            pts1_b.append(kp2[m[0].queryIdx].pt)
        
        pts1_arr = np.asarray(pts1)
        pts2_arr = np.asarray(pts2)
        pts_12 = np.hstack((pts1_arr, pts2_arr))
        pts1_arr_b = np.asarray(pts1_b)
        pts2_arr_b = np.asarray(pts2_b)        
        pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))
           
        pts1_ts = []
        pts2_ts = []        
        for pts in pts_12:
            pts_comp = np.asarray(pts, dtype = np.int)
            for pts_b in pts_21:
                pts_b_comp = np.asarray(pts_b, dtype = np.int)
                if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                    and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                    pts1_ts.append(pts[0:2].tolist())
                    pts2_ts.append(pts[2:4].tolist())                
                    break
        
        pts1 = np.asarray(pts1_ts, dtype=np.float32)
        pts2 = np.asarray(pts2_ts, dtype=np.float32)
        
        #print('Matches twosided calculated')
            
        return pts1, pts2
    
    #detect Harris corner features
    def HarrisCorners(self, image_file, kp_nbr=None, visualize=False, img_import=False):
        
        if img_import:
            image_gray = image_file
        else:
            image = cv2.imread(image_file)
            image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                                                             
        image_gray = np.uint8(image_gray) 
        
        '''detect Harris corners'''
        keypoints = cv2.cornerHarris(image_gray,2,3,0.04)
        keypoints = cv2.dilate(keypoints,None)                                                              
        
        #reduce keypoints to specific number
        thresh_kp_reduce = 0.01
        keypoints_prefilt = keypoints
        keypoints = np.argwhere(keypoints > thresh_kp_reduce * keypoints.max())
    
        if not kp_nbr==None:
            keypoints_reduced = keypoints
            while len(keypoints_reduced) >= kp_nbr:
                thresh_kp_reduce = thresh_kp_reduce + 0.01
                keypoints_reduced = np.argwhere(keypoints_prefilt > thresh_kp_reduce * keypoints_prefilt.max())
        else:
            keypoints_reduced = keypoints       
        
        if visualize:
            drawing = drw.Drawing()
            drawing.plot_harris_points(image, keypoints_reduced)
            
        keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints_reduced]
        
        #print('number of Harris corners:' + str(len(keypoints)))
        
        return keypoints, keypoints_reduced #keypoints_reduced for drawing
    
    #calculate ORB descriptors at detected features (using various feature detectors)
    def OrbDescriptors(self, image_file, keypoints):
        image = cv2.imread(image_file)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                                             
        image_gray = np.uint8(image_gray) 
        
        '''perform ORB'''
        orb = cv2.ORB()
        keypoints, descriptors = orb.compute(image_gray, keypoints)
        
        return keypoints, descriptors
    
    #calculate SIFT descriptors at detected features (using various feature detectors)
    def SiftDescriptors(self, image_file, keypoints):
        image = cv2.imread(image_file)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                                             
        image_gray = np.uint8(image_gray) 
            
        '''perform SIFT'''
        siftCV2 = cv2.SIFT()
        keypoints, descriptors = siftCV2.compute(image_gray, keypoints)
        descriptors = descriptors.astype(np.uint8)
        
        return keypoints, descriptors
    
    #match SIFT features using SIFT matching
    def match_SIFT(self, desc1, desc2):
        '''For each descriptor in the first image, select its match in the second image.
        input: desc1 (descriptors for the first image),
        desc2 (same for the second image).'''
        
        desc1 = np.array([d/plt.linalg.norm(d) for d in desc1])
        desc2 = np.array([d/plt.linalg.norm(d) for d in desc2])
        
        dist_ratio = 0.6
        desc1_size = desc1.shape
        
        matchscores = np.zeros((desc1_size[0],1),'int')
        desc2t = desc2.T #precompute matrix transpose
        for i in range(desc1_size[0]):
            dotprods = np.dot(desc1[i,:], desc2t)   #vector of dot products
            dotprods = 0.9999*dotprods
            # inverse cosine and sort, return index for features in second Image
            indx = np.argsort(plt.arccos(dotprods))
            
            # check if nearest neighbor has angle less than dist_ratio times 2nd
            if plt.arccos(dotprods)[indx[0]] < dist_ratio * plt.arccos(dotprods)[indx[1]]:
                matchscores[i] = np.int(indx[0])
                
        return matchscores
    
    #match SIFT features using SIFT matching and perform two-sided
    def match_twosided_SIFT(self, desc1, desc2):
        '''Two-sided symmetric version of match().'''
        
        matches_12 = self.match_SIFT(desc1,desc2)
        matches_21 = self.match_SIFT(desc2,desc1)
        
        ndx_12 = matches_12.nonzero()[0]
        
        # remove matches that are not symmetric
        for n in ndx_12:
            if matches_21[int(matches_12[n])] != n:
                matches_12[n] = 0
                
        return matches_12