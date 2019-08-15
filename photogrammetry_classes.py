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
import sys, csv, os
import pandas as pd
import scipy.spatial
import cv2
import math, numpy as np
import matplotlib.pyplot as plt
import matplotlib


#drop duplicate 3D points
def drop_dupl(x,y,z):
    df = pd.DataFrame({'x':x, 'y':y, 'z':z})
    dupl_dropped = df.drop_duplicates(cols=['x', 'y', 'z'])    
    return np.asarray(dupl_dropped)

#drop duplicate 2D points
def drop_dupl_xy(x,y):
    df = pd.DataFrame({'x':x, 'y':y})
    dupl_dropped = df.drop_duplicates(cols=['x', 'y'])    
    return np.asarray(dupl_dropped)


#class of interior camera geometry
class camera_interior:

    #interior geometry parameters: principle point, focal length, distortion parameters, sensor information
    def __init__(self):
        self.xh = 0
        self.yh = 0
        self.ck = None  #focal length
        self.A1 = 0 #radial distortion
        self.A2 = 0
        self.A3 = 0
        self.B1 = 0 #tangential distortion
        self.B2 = 0
        self.C1 = 0 #skew
        self.C2 = 0
        self.resolution_x = None        
        self.resolution_y = None
        self.sensor_size_x = None
        self.sensor_size_y = None
        self.r0 = 0
    
    
    #read camera parameters from AICON file (specific format)
    def read_aicon_ior(self, directory, ior_file=None):
        #read aicon interior geometry in mm
        if ior_file == None:    #names in one txt
            file_read = open(directory)
        else:   #names in two separate txt
            file_read = open(os.path.join(directory, ior_file))
            
        ior_table = file_read.read().split(' ')      #normals created in CC
        file_read.close()
        
        self.ck = np.float(ior_table[2])
        self.xh = np.float(ior_table[3])
        self.yh = np.float(ior_table[4])
        self.A1 = np.float(ior_table[5])
        self.A2 = np.float(ior_table[6])
        self.A3 = np.float(ior_table[8])
        self.r0 = np.float(ior_table[7])
        self.B1 = np.float(ior_table[9])
        self.B2 = np.float(ior_table[10])
        self.C1 = np.float(ior_table[11])
        self.C2 = np.float(ior_table[12])
        self.sensor_size_x = np.float(ior_table[13])
        self.sensor_size_y = np.float(ior_table[14])
        self.resolution_x = np.float(ior_table[15])
        self.resolution_y = np.float(ior_table[16])


class Pt3D:
    
    #3D point (can include RGB information)
    def __init__(self):

        self.X = None
        self.Y = None
        self.Z = None

        self.R = None
        self.G = None
        self.B = None  
        
        self.rgb = False
    
    #assign coordinates to 3D point    
    def read_imgPts_3D(self, pts_3D):
        
        self.X = pts_3D[:,0]
        self.Y = pts_3D[:,1]
        self.Z = pts_3D[:,2]
        
        if self.rgb == True:
            self.R = pts_3D[:,3]
            self.G = pts_3D[:,4]
            self.B = pts_3D[:,5]                   
                

class PtImg:
    
    #2D point
    def __init__(self):
        
        self.x = None
        self.y = None
    
    #assign coordinates to 2D point        
    def read_imgPts(self, img_pts):
        
        self.x = img_pts[:,0]
        self.y = img_pts[:,1]
        

#perform image measurements
class image_measures:

    def __init__(self):
        pass
    
    #convert pixel coordinate into metric image coordinates
    def pixel_to_metric(self, img_pts, interior_orient):
        
        center_x = interior_orient.resolution_x/2 + 0.5
        center_y = interior_orient.resolution_y/2 + 0.5
        pixel_size = interior_orient.sensor_size_x/interior_orient.resolution_x
        
        pixel_size_control = interior_orient.sensor_size_y/interior_orient.resolution_y
        if not pixel_size > (pixel_size_control - pixel_size * 0.1) and pixel_size < (pixel_size_control + pixel_size * 0.1):
            sys.exit('error with pixel size: x not equal y')        
        
        img_pts_mm = PtImg()
        img_pts_mm.x = np.asarray((img_pts.x - 0.5 - center_x) * pixel_size)
        img_pts_mm.y = np.asarray((img_pts.y - 0.5 - center_y) * (-1 * pixel_size))
        
        return img_pts_mm
    
    #convert metric image coordinates into pixel coordinates
    def metric_to_pixel(self, img_pts, interior_orient):
    
        pixel_size = interior_orient.sensor_size_x/interior_orient.resolution_x
        
        pixel_size_control = interior_orient.sensor_size_y/interior_orient.resolution_y
        if not pixel_size > (pixel_size_control - pixel_size * 0.1) and pixel_size < (pixel_size_control + pixel_size * 0.1):
            sys.exit('error with pixel size: x not equal y')    
        
        img_pts_pix = PtImg()
        img_pts_pix.x = img_pts.x / pixel_size + np.ones(img_pts.x.shape[0]) * (interior_orient.resolution_x/2)
        img_pts_pix.y = interior_orient.resolution_y - (img_pts.y / pixel_size + np.ones(img_pts.y.shape[0]) * (interior_orient.resolution_y/2))
        
        return img_pts_pix    
            
    #undistort image measurements considering interior camera geometry (using AICON model)  
    def undistort_img_coos(self, img_pts, interior_orient, mm_val=False):
    # source code from Hannes Sardemann rewritten for Python
        #img_pts: array with x and y values in pixel (if in mm state this, so can be converted prior to pixel)
        #interior_orient: list with interior orientation parameters in mm
        #output: in mm
        
        ck = -1 * interior_orient.ck
    
        #transform pixel values into mm-measurement
        if mm_val == False:    
            img_pts = self.pixel_to_metric(img_pts, interior_orient)
            
        x_img = img_pts.x
        y_img = img_pts.y
    
        x_img_1 = img_pts.x
        y_img_1 = img_pts.y
        
        
        #start iterative undistortion
        iteration = 0
        
        test_result = [10, 10]
        
        while np.max(test_result) > 1e-14:
            
            if iteration > 1000:
                sys.exit('No solution for un-distortion')
                break
            
            iteration = iteration + 1
            
            camCoo_x = x_img
            camCoo_y = y_img
            
            if interior_orient.r0 == 0:
                x_dash = camCoo_x / (-1 * ck)
                y_dash = camCoo_y / (-1 * ck)
                r2 = x_dash**2 + y_dash**2  #img radius
            else:
                x_dash = camCoo_x
                y_dash = camCoo_y
                if x_dash.shape[0] < 2:
                    r2 = np.float(x_dash**2 + y_dash**2)  #img radius
                else:
                    r2 = x_dash**2 + y_dash**2
                r = np.sqrt(r2)  
                    
            '''extended Brown model'''        
            #radial distoriton   
            if interior_orient.r0 == 0:
                p1 = ((interior_orient.A3 * r2 + (np.ones(r2.shape[0]) * interior_orient.A2)) * r2 + (np.ones(r2.shape[0]) * interior_orient.A1)) * r2            
            else:
                p1 = (interior_orient.A1 * (r**2 - (interior_orient.r0**2)) + interior_orient.A2 * (r**4 - interior_orient.r0**4) + 
                      interior_orient.A3 * (r**6 - interior_orient.r0**6))
                
            dx_rad = x_dash * p1
            dy_rad = y_dash * p1
            
            
            #tangential distortion
            dx_tan = (interior_orient.B1 * (r2 + 2 * x_dash**2)) + 2 * interior_orient.B2 * x_dash * y_dash
            dy_tan = (interior_orient.B2 * (r2 + 2 * y_dash**2)) + 2 * interior_orient.B1 * x_dash * y_dash
            
            
            #combined distortion
            dx = dx_rad + dx_tan
            dy = dy_rad + dy_tan
            
            x_roof = x_dash + dx
            y_roof = y_dash + dy
            
            
            #adding up distortion to recent distorted coordinate
            if interior_orient.r0 == 0:
                x_img_undistort = np.ones(x_dash.shape[0]) * interior_orient.xh - ck * (np.ones(x_roof.shape[0]) + interior_orient.C1) * x_roof - ck * interior_orient.C2 * y_roof
                y_img_undistort = np.ones(y_roof.shape[0]) * interior_orient.yh - ck * y_roof
            else:
                x_img_undistort = np.ones(x_dash.shape[0]) * interior_orient.xh + (np.ones(x_roof.shape[0]) + interior_orient.C1) * x_roof + interior_orient.C2 * y_roof
                y_img_undistort = np.ones(y_roof.shape[0]) * interior_orient.yh + y_roof
                
            
            #subtracting distortion from original coordinate
            x_img = x_img_1 - (x_img_undistort - x_img)
            y_img = y_img_1 - (y_img_undistort - y_img)
            
            
            #test result if difference between re-distorted (undistorted) coordinates fit to original img coordinates
            test_result[0] = np.max(np.abs(x_img_undistort - img_pts.x))
            test_result[1] = np.max(np.abs(y_img_undistort - img_pts.y))
        

        img_pts_undist = PtImg() 
        img_pts_undist.x = x_img
        img_pts_undist.y = y_img
        
        return img_pts_undist   #in mm
    
    #undistort image measurements considering interior camera geometry (using Agisoft PhotoScan model)  
    def undistort_img_coos_Agisoft(self, img_pts, interior_orient, mm_val=False):
    # source code from Hannes Sardemann rewritten for Python
        #img_pts: array with x and y values in pixel (if in mm state this, so can be converted prior to pixel)
        #interior_orient: list with interior orientation parameters in mm
        #output: in mm
        
        ck = -1 * interior_orient.ck
    
        #transform pixel values into mm-measurement
        if mm_val == False:    
            img_pts = self.pixel_to_metric()
            
        x_img = img_pts[:,0]
        y_img = img_pts[:,1]
    
        x_img_1 = img_pts[:,0]
        y_img_1 = img_pts[:,1]
        
        
        #start iterative undistortion
        iteration = 0
        
        test_result = [10, 10]
        
        while np.max(test_result) > 1e-14:
            
            if iteration > 1000:
                sys.exit('No solution for un-distortion')
                break
            
            iteration = iteration + 1
            
            camCoo_x = x_img
            camCoo_y = y_img
            
            if self.interior_orient.r0 == 0:
                x_dash = camCoo_x / (-1 * ck)
                y_dash = camCoo_y / (-1 * ck)
                r2 = x_dash**2 + y_dash**2  #img radius
            else:
                x_dash = camCoo_x
                y_dash = camCoo_y
                if x_dash.shape[0] < 2:
                    r2 = np.float(x_dash**2 + y_dash**2)  #img radius
                else:
                    r2 = x_dash**2 + y_dash**2
                r = np.sqrt(r2)  
                    
            '''extended Brown model'''        
            #radial distoriton   
            if self.interior_orient.r0 == 0:
                p1 = ((interior_orient.A3 * r2 + (np.ones(r2.shape[0]) * interior_orient.A2)) * r2 + (np.ones(r2.shape[0]) * interior_orient.A1)) * r2            
            else:
                p1 = (interior_orient.A1 * (r**2 - (interior_orient.r0**2)) + interior_orient.A2 * (r**4 - interior_orient.r0**4) + 
                      interior_orient.A3 * (r**6 - interior_orient.r0**6))
                
            dx_rad = x_dash * p1
            dy_rad = y_dash * p1
            
            
            #tangential distortion
            dx_tan = (interior_orient.B1 * (r2 + 2 * x_dash**2)) + 2 * interior_orient.B2 * x_dash * y_dash
            dy_tan = (interior_orient.B2 * (r2 + 2 * y_dash**2)) + 2 * interior_orient.B1 * x_dash * y_dash
            
            
            #combined distortion
            dx = dx_rad + dx_tan
            dy = dy_rad + dy_tan
            
            x_roof = x_dash + dx
            y_roof = y_dash + dy
            
            
            #adding up distortion to recent distorted coordinate
            if self.interior_orient.r0 == 0:
                x_img_undistort = np.ones(x_dash.shape[0]) * interior_orient.xh - ck * (np.ones(x_roof.shape[0]) + interior_orient.C1) * x_roof - ck * interior_orient.C2 * y_roof
                y_img_undistort = np.ones(y_roof.shape[0]) * interior_orient.yh - ck * y_roof
            else:
                x_img_undistort = np.ones(x_dash.shape[0]) * interior_orient.xh + (np.ones(x_roof.shape[0]) + interior_orient.C1) * x_roof + interior_orient.C2 * y_roof
                y_img_undistort = np.ones(y_roof.shape[0]) * interior_orient.yh + y_roof
                
            
            #subtracting distortion from original coordinate
            x_img = x_img_1 - (x_img_undistort - x_img)
            y_img = y_img_1 - (y_img_undistort - y_img)
            
            
            #test result if difference between re-distorted (undistorted) coordinates fit to original img coordinates
            test_result[0] = np.max(np.abs(x_img_undistort - img_pts[:,0]))
            test_result[1] = np.max(np.abs(y_img_undistort - img_pts[:,1]))
        
        
        x_undistort = x_img #in mm
        y_undistort = y_img #in mm
        
        
        x_undistort = x_undistort.reshape(x_undistort.shape[0],1)
        y_undistort = y_undistort.reshape(y_undistort.shape[0],1)
        img_pts_undist = np.hstack((x_undistort, y_undistort))
        
        return img_pts_undist   #in mm


#convert 2D measurements to 3D coordinates
class TwoD_to_ThreeD:
    
    
    def __init__(self):
        pass
    
    #help class to assign image coordinates to object coordinates based on same ID
    class AssignedCoo:
        
        def __init__(self):
            
            self.x = []
            self.y = []
            self.X = []
            self.Y = []
            self.Z = []
        
        #array with assigned image coordinates 
        def mat_assignedCoo_img(self, x, y):
                       
            matAssCoo_img_x = np.asarray(x)
            matAssCoo_img_y = np.asarray(y)
            
            matAssCoo_img = np.hstack((matAssCoo_img_x.reshape(matAssCoo_img_x.shape[0],1), 
                                       matAssCoo_img_y.reshape(matAssCoo_img_y.shape[0],1)))
            
            return matAssCoo_img
        
        #array with assigned object coordinates 
        def mat_assignedCoo_obj(self, X, Y, Z):
                        
            matAssCoo_obj_X = np.asarray(X)
            matAssCoo_obj_Y = np.asarray(Y)
            matAssCoo_obj_Z = np.asarray(Z)
            
            matAssCoo_obj = np.hstack((matAssCoo_obj_X.reshape(matAssCoo_obj_X.shape[0],1), 
                                       matAssCoo_obj_Y.reshape(matAssCoo_obj_Y.shape[0],1)))
            matAssCoo_obj = np.hstack((matAssCoo_obj, 
                                       matAssCoo_obj_Z.reshape(matAssCoo_obj_Z.shape[0],1)))            
            
            return matAssCoo_obj
        
        #array with assigned image and object coordinates  
        def mat_assignedCoo_all(self, x, y, X, Y, Z):
    
            matAssCoo_img = self.mat_assignedCoo_img(x, y)
            matAssCoo_obj = self.mat_assignedCoo_obj(X, Y, Z)
            matAssCoo_all = np.hstack((matAssCoo_img, matAssCoo_obj))
            
            return matAssCoo_all
    
    #function to assigne corresponding coordinates from image measurement to object points (based on ID)
    def assign_ImgToObj_Measurement(self, obj_pts, img_pts):
    #obj_pts: object coordinate (ID, X, Y, Z)
    #img_pts: image coordinates (ID, x, y)
    
        img_gcp_coos = self.AssignedCoo()
        # img_coos = []
        # gcp_coos = []
        pt_id = []
        nbr_rows = 0
        for row_gcp in obj_pts:
            for row_pts in img_pts:
                if row_gcp[0] == row_pts[0]:
                    img_gcp_coos.x.append(row_pts[1])
                    img_gcp_coos.y.append(row_pts[2])                    
                    img_gcp_coos.X.append(row_gcp[1])
                    img_gcp_coos.Y.append(row_gcp[2])
                    img_gcp_coos.Z.append(row_gcp[3])
                    
                    pt_id.append(row_pts[0])
                    nbr_rows = nbr_rows + 1
                    break       
        
        return img_gcp_coos, pt_id
  
    #perform exterior calibration (orient image) using RANSAC model to detect outliers in corresponding (assigned) 
    #image and object points
    #solvePNP from openCV is used to estimate exterior geometry
    def image_orientation_RANSAC(self, img_gcp_coos, cam_file_opencv, reprojectionError=5):     #register_frame
        #cam_file_opencv: interior camera parameters in pixel
       
        '''read camera file with interior orientation information'''   
        #transform metric values to pixel values
        ck, cx, cy, k1, k2, k3, p1, p2 = cam_file_opencv
    
        ''' give information about interior camera geometry'''
        #camera matrix opencv
        camMatrix = np.zeros((3,3),dtype=np.float32)
        camMatrix[0][0] = ck
        camMatrix[0][2] = cx
        camMatrix[1][1] = ck
        camMatrix[1][2] = cy
        camMatrix[2][2] = 1.0           
        distCoeff = np.asarray([k1, k2, p1, p2, k3], dtype=np.float32)                   
        
        assCoo = self.AssignedCoo()
        gcp_coos = assCoo.mat_assignedCoo_obj(img_gcp_coos.X, img_gcp_coos.Y, img_gcp_coos.Z)
        img_pts = assCoo.mat_assignedCoo_img(img_gcp_coos.x, img_gcp_coos.y)
        
                
        '''resolve for exterior camera parameters'''
        #solve for exterior orientation
        rvec_solved, tvec_solved, inliers = cv2.solvePnPRansac(gcp_coos, img_pts, camMatrix, distCoeff, reprojectionError) # iterationsCount=2000, reprojectionError=5
    #     if not inliers == None:
    #         print('numer of used points for RANSAC PNP: ' + str(len(inliers)))
    
    #     _, rvec_solved, tvec_solved = cv2.solvePnP(gcp_coos, img_pts, camMatrix, distCoeff,
    #                                                rvec_solved, tvec_solved, useExtrinsicGuess=True)
        
        '''convert to angles and XYZ'''
        np_rodrigues = np.asarray(rvec_solved[:,:],np.float64)
        rot_matrix = cv2.Rodrigues(np_rodrigues)[0]
        
        position = -np.matrix(rot_matrix).T * np.matrix(tvec_solved) 
            
        return rot_matrix, position, inliers    

    #convert point coordinates from 3D point class into array
    def coos_to_mat(self, point_cloud):
 
        point_cloudXYZ = np.hstack((point_cloud.X.reshape(point_cloud.X.shape[0],1), point_cloud.Y.reshape(point_cloud.Y.shape[0],1)))
        point_cloudXYZ = np.hstack((point_cloudXYZ, point_cloud.Z.reshape(point_cloud.Z.shape[0],1)))
        
        return point_cloudXYZ
        
    #convert point RGB values from 3D point class into array    
    def rgb_to_mat(self, point_cloud):
        
        point_cloudRGB = np.hstack((point_cloud.R.reshape(point_cloud.R.shape[0],1), point_cloud.G.reshape(point_cloud.G.shape[0],1)))
        point_cloudRGB = np.hstack((point_cloudRGB, point_cloud.B.reshape(point_cloud.B.shape[0],1)))    
          
        return point_cloudRGB
    
    #transform point cloud from object space into image space
    def pointCl_to_Img(self, point_cloud, eor_mat):
        
        point_cloudXYZ = self.coos_to_mat(point_cloud)
        if point_cloud.rgb:
            point_cloudRGB = self.rgb_to_mat(point_cloud)
        
        point_cloud_trans = np.matrix(np.linalg.inv(eor_mat)) * np.matrix(np.vstack((point_cloudXYZ.T, np.ones(point_cloudXYZ.shape[0]))))
        point_cloud_trans = point_cloud_trans.T
                
        if point_cloud.rgb:
            point_cloud_trans_rgb = np.hstack((point_cloud_trans, point_cloudRGB))
            point_cloud_img = Pt3D()
            point_cloud_img.read_imgPts_3D(point_cloud_trans_rgb)           
        else:        
            point_cloud_img = Pt3D()
            point_cloud_img.read_imgPts_3D(point_cloud_trans)
        
        return point_cloud_img

    #project 3D point cloud into image space
    def project_pts_into_img(self, eor_mat, ior_mat, point_cloud, plot_results=False, neg_x=False):
        #point cloud including RGB
        #ior_mat from read_aicon_ior
                
        
        '''transform point cloud into camera coordinate system'''
        point_cloud = self.pointCl_to_Img(point_cloud, eor_mat)

        
        #remove points behind the camera
        if point_cloud.rgb:
            df_points = pd.DataFrame(np.hstack((self.coos_to_mat(point_cloud), self.rgb_to_mat(point_cloud))))
        else:
            df_points = pd.DataFrame(self.coos_to_mat(point_cloud))
        df_points = df_points.loc[df_points[2] > 0] 
        
        pt3D = Pt3D()
        pt3D.read_imgPts_3D(np.asarray(df_points))

        del df_points
        
        
        '''inbetween coordinate system'''
        x = pt3D.X / pt3D.Z
        y = pt3D.Y / pt3D.Z
        d = pt3D.Z
        
        if neg_x:
            ptCloud_img = np.hstack((x.reshape(x.shape[0],1)*-1, y.reshape(y.shape[0],1)))
        else:
            ptCloud_img = np.hstack((x.reshape(x.shape[0],1), y.reshape(y.shape[0],1)))
        ptCloud_img = np.hstack((ptCloud_img, d.reshape(d.shape[0],1)))
        if not ptCloud_img.shape[0] > 0:    #take care if img registration already erroneous
            return None
        
        if point_cloud.rgb:
            ptCloud_img = np.hstack((ptCloud_img, self.rgb_to_mat(pt3D)))
        
        pt3D.read_imgPts_3D(ptCloud_img)
        ptCloud_img = pt3D
    
        if plot_results:
            if point_cloud.shape[1] > 3:
                rgb = self.rgb_to_mat(ptCloud_img) / 256
            _, ax = plt.subplots()
            if point_cloud.rgb:
                ax.scatter(x, y, s=5, edgecolor=None, lw = 0, facecolors=rgb)
            else:
                ax.scatter(x, y, s=5, edgecolor=None, lw = 0)
            plt.title('3D point cloud in image space')
            plt.show()
        
    #     #remove points outside field of view
    #     test1 = np.abs(point_cloud[:,0]) > np.abs((ior_mat.resolution_x - ior_mat.xh) / (-1*ior_mat.ck) * point_cloud[:,2])
    #     test2 = np.abs(point_cloud[:,1]) > np.abs((ior_mat.resolution_y - ior_mat.yh) / (ior_mat.ck) * point_cloud[:,2])
    #     test = np.where(np.logical_and(test1 == True, test2 == True))    
    #     ptCloud_img = ptCloud_img[test]   
    
      
        '''calculate depth map but no interpolation (solely for points from point cloud'''
        ptCloud_img_proj = PtImg()
        ptCloud_img_proj.x = ptCloud_img.X * -1 * ior_mat.ck
        ptCloud_img_proj.y = ptCloud_img.Y * ior_mat.ck

        img_measure = image_measures()
        ptCloud_img_px = img_measure.metric_to_pixel(ptCloud_img_proj, ior_mat)
        z_vals = ptCloud_img.Z
        
        ptCloud_img_px_depth = Pt3D
        ptCloud_img_px_depth.X = ptCloud_img_px.x
        ptCloud_img_px_depth.Y = ptCloud_img_px.y
        ptCloud_img_px_depth.Z = z_vals        
        
        if point_cloud.rgb:
            ptCloud_img_px_depth.R = ptCloud_img.R
            ptCloud_img_px_depth.G = ptCloud_img.G
            ptCloud_img_px_depth.B = ptCloud_img.B 

              
        return ptCloud_img_px_depth
    
    #find nearest neighbors between reference point cloud (3D point cloud project into image space) and
    #target points (image points of water line) 
    def NN_pts(self, reference_pts, target_pts, max_NN_dist=1, plot_results=False,
               closest_to_cam=False, ior_mat=None, eor_mat=None):    
        
        reference_pts_xyz = np.hstack((reference_pts.X.reshape(reference_pts.X.shape[0],1), 
                                          reference_pts.Y.reshape(reference_pts.Y.shape[0],1)))
        reference_pts_xyz = np.hstack((reference_pts_xyz, reference_pts.Z.reshape(reference_pts.Z.shape[0],1)))
        reference_pts_xy_int = np.asarray(reference_pts_xyz[:,0:2], dtype = np.int)
        
        targ_x = np.asarray(target_pts.x, dtype = np.int)
        targ_y = np.asarray(target_pts.y, dtype = np.int)
        target_pts_int = np.hstack((targ_x.reshape(targ_x.shape[0],1), targ_y.reshape(targ_y.shape[0],1)))
        
        points_list = list(target_pts_int)
    
        #define kd-tree
        mytree = scipy.spatial.cKDTree(reference_pts_xy_int)
    #    dist, indexes = mytree.query(points_list)
    #    closest_ptFromPtCloud = reference_pts[indexes,0:3]
        
        #search for nearest neighbour
        indexes = mytree.query_ball_point(points_list, max_NN_dist)   #find points within specific distance (here in pixels)
        
        #filter neighbours to keep only point closest to camera if several NN found
        NN_points_start = True
        NN_skip = 0
        NN_points = None
        dist_to_pz_xy = None
        
        nearestPtsToWaterPt_xyz = Pt3D()
        nearestPtsToWaterPt_xy = PtImg()
        
        for nearestPts_ids in indexes:
            if not nearestPts_ids:  #if no nearby point found, skip
                NN_skip = NN_skip + 1
                continue
            
            #select all points found close to waterline point
            nearestPtsToWaterPt_d = reference_pts_xyz[nearestPts_ids,0:3]
            
            if closest_to_cam:
                nearestPtsToWaterPt_xyz.read_imgPts_3D(nearestPtsToWaterPt_d)
                nearestPtsToWaterPt_xy.read_imgPts(nearestPtsToWaterPt_d) 
                
                '''select only point closest to camera'''             
                #transform image measurement into object space
                img_measure = image_measures()
                imgPts_mm = img_measure.pixel_to_metric(nearestPtsToWaterPt_xy, ior_mat)        
                nearestPtsToWaterPt_xyz.X = imgPts_mm.x
                nearestPtsToWaterPt_xyz.Y = imgPts_mm.y
                
                xyd_map_mm = self.imgDepthPts_to_objSpace(nearestPtsToWaterPt_xyz, eor_mat, ior_mat)
                xyd_map_mm = drop_dupl(xyd_map_mm.X, xyd_map_mm.Y, xyd_map_mm.Z)
                  
                #calculate shortest distance to camera centre
                pz_coo = Pt3D()
                pz_coo.read_imgPts_3D(eor_mat[0:3,3])                
                dist_to_pz = np.sqrt(np.square(pz_coo.X - xyd_map_mm.X) + np.square(pz_coo.Y - xyd_map_mm.Y) + 
                                     np.square(pz_coo.Z - xyd_map_mm.Z))
                
                xyd_map_mm = self.coos_to_mat(xyd_map_mm)                
                dist_to_pz_xy = np.hstack((xyd_map_mm, dist_to_pz.reshape(dist_to_pz.shape[0],1)))
                dist_to_pz_xy_df = pd.DataFrame(dist_to_pz_xy)                
                closest_pt_to_cam = dist_to_pz_xy_df.loc[dist_to_pz_xy_df[3].idxmin()]
                closestCameraPt = np.asarray(closest_pt_to_cam)
    
            df_nearestPtsToWaterPt_d = pd.DataFrame(nearestPtsToWaterPt_d)        
            id_df_nearestPtsToWaterPt_d = df_nearestPtsToWaterPt_d.loc[df_nearestPtsToWaterPt_d[2].idxmin()]
            closestCameraPt = np.asarray(id_df_nearestPtsToWaterPt_d)
            
            if NN_points_start:
                NN_points_start = False
                NN_points = closestCameraPt.reshape(1, closestCameraPt.shape[0])
            else:
                NN_points = np.vstack((NN_points, closestCameraPt.reshape(1,closestCameraPt.shape[0])))               
    
    
        print('NN skipped: ' + str(NN_skip))
        
    #     if dist_to_pz_xy == None:
    #         return NN_points, None, None
    
        if NN_points  == None:
            NN_points_xyz = None
        else:
            NN_points_xyz = Pt3D()
            NN_points_xyz.read_imgPts_3D(NN_points)
    
        return NN_points_xyz    #, np.min(dist_to_pz_xy[:,2]), np.max(dist_to_pz_xy[:,2])
    
    #convert 3D points in image space into object space
    def imgDepthPts_to_objSpace(self, img_pts_xyz, eor_mat, ior_mat):
            
        '''calculate inbetween coordinate system'''
        img_pts_xyz.X = img_pts_xyz.X / (-1 * ior_mat.ck)
        img_pts_xyz.Y = img_pts_xyz.Y / ior_mat.ck
        
        img_pts_xyz.X = img_pts_xyz.X * img_pts_xyz.Z
        img_pts_xyz.Y = img_pts_xyz.Y * img_pts_xyz.Z
        
        imgPts_xyz = self.coos_to_mat(img_pts_xyz)
        
        '''transform into object space'''
        imgPts_XYZ = np.matrix(eor_mat) * np.matrix(np.vstack((imgPts_xyz.T, np.ones(imgPts_xyz.shape[0])))) 
        imgPts_XYZ = np.asarray(imgPts_XYZ.T)
        
        imgPts_XYZ_out = Pt3D()
        imgPts_XYZ_out.read_imgPts_3D(imgPts_XYZ)
        
        return imgPts_XYZ_out
    

#various conversion tools
class conversions:
    
    def __init__(self):
        pass
    
    #convert openCV rotation matrix into Euler angles
    def rotMat_to_angle(self, rot_mat, position):
        multipl_array = np.array([[1,0,0],[0,-1,0],[0,0,1]])  
        rot_matrix = -1 * (np.matrix(rot_mat) * np.matrix(multipl_array))
        rot_matrix = np.asarray(rot_matrix)
        omega, phi, kappa = self.rot_matrix_to_euler(rot_matrix, 'radian')
        rotation = np.asarray([omega, phi, -1*kappa]) #note that kappa needs to be multiplied with -1 to rotate correctly  
        exterior_approx = np.vstack((position.reshape(position.shape[0],1), rotation.reshape(rotation.shape[0],1)))
    
        return exterior_approx
    
    
    #convert Euler angles into rotation matrix
    def rot_Matrix(self, omega,phi,kappa,unit='grad'):        #radians
    # unit: rad = radians, gon, grad
        # gon to radian
        if unit == 'gon':
            omega = omega * (math.pi/200)
            phi = phi * (math.pi/200)
            kappa = kappa * (math.pi/200)
         
        # grad to radian
        elif unit == 'grad':
            omega = omega * (math.pi/180)
            phi = phi * (math.pi/180)
            kappa = kappa * (math.pi/180)
        
        # radian    
        elif unit == 'rad':
            omega = omega
            phi = phi
            kappa = kappa
        
        r11 = math.cos(phi) * math.cos(kappa)
        r12 = -math.cos(phi) * math.sin(kappa)
        r13 = math.sin(phi)
        r21 = math.cos(omega) * math.sin(kappa) + math.sin(omega) * math.sin(phi) * math.cos(kappa)
        r22 = math.cos(omega) * math.cos(kappa) - math.sin(omega) * math.sin(phi) * math.sin(kappa)
        r23 = -math.sin(omega) * math.cos(phi)
        r31 = math.sin(omega) * math.sin(kappa) - math.cos(omega) * math.sin(phi) * math.cos(kappa)
        r32 = math.sin(omega) * math.cos(kappa) + math.cos(omega) * math.sin(phi) * math.sin(kappa)
        r33 = math.cos(omega) * math.cos(phi)
        
        rotMat = np.array(((r11,r12,r13),(r21,r22,r23),(r31,r32,r33)))        
        return rotMat
    
    
    #convert photogrammetric rotation matrix into Euler angles
    def rot_matrix_to_euler(self, R, unit='grad'):
        y_rot = math.asin(R[2][0]) 
        x_rot = math.acos(R[2][2]/math.cos(y_rot))    
        z_rot = math.acos(R[0][0]/math.cos(y_rot))
        if unit == 'grad':
            y_rot_angle = y_rot *(180/np.pi)
            x_rot_angle = x_rot *(180/np.pi)
            z_rot_angle = z_rot *(180/np.pi)    
        else: #unit is radiant
            y_rot_angle = y_rot
            x_rot_angle = x_rot
            z_rot_angle = z_rot     
        return x_rot_angle,y_rot_angle,z_rot_angle  #omega, phi, kappa


#perform resection with adjustment to orient camera using collinearity equations
# source code for least square adjustment from Danilo Schneider rewritten for Python
class resection:
    
    def __init__(self):
        pass


    #generate observation vector for least squares adjustment
    def l_vector_resection(self, ImgCoos_GCPCoos, camera_interior, camera_exterior):
    #ImgCoos_GCPCoos: assigned image coordinates and object coordinates of ground control points 
    # (numpy array [x_vec, y_vec, X_vec, Y_vec, Z_vec])
    
        l_vec = np.zeros((2*ImgCoos_GCPCoos.shape[0],1))
        
        i = 0
        for point in ImgCoos_GCPCoos:  
            x, y = self.model_resection(camera_interior, point[2:5], camera_exterior)
            l_vec[2*i] = point[0]-x
            l_vec[2*i+1] = point[1]-y  
            
            i = i + 1
            
        return l_vec
    
    
    #generate design matrix for least squares adjustment
    def A_mat_resection(self, ImgCoos_GCPCoos, camera_exterior, camera_interior, e=0.0001, param_nbr=6):
    #ImgCoos_GCPCoos: assigned image coordinates and object coordinates of ground control points 
    #camera_exterior: coordinates of projection centre  and angles of ration matrix (numpy array [X0, Y0, Z0, omega, phi, kappe])
    #camera_interior: interior camera orientation (numpy array [ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0]), Brown (aicon) model
    #param_nbr: define number of parameters, which are adjusted (standard case only XYZ, OmegaPhiKappa)
    #e: epsilon
       
        #generates empty matrix
        A = np.zeros((2*ImgCoos_GCPCoos.shape[0], param_nbr))
        
        #fills design matrix
        camera_exterior = camera_exterior.reshape(camera_exterior.shape[0],1)
        i = 0
        for point in ImgCoos_GCPCoos:     
            for j in range(param_nbr):
                #numerical adjustment (mini distance above and below point to estimate slope)
                parameter1 = np.zeros((camera_exterior.shape[0],1))
                parameter2 = np.zeros((camera_exterior.shape[0],1))
                parameter1[:] = camera_exterior[:]
                parameter1[j] = camera_exterior[j] - e
                parameter2[:] = camera_exterior[:]
                parameter2[j] = camera_exterior[j] + e
                
                x2, y2 = self.model_resection(camera_interior, point[2:5], parameter2)
                x1, y1 = self.model_resection(camera_interior, point[2:5], parameter1)
                
                A[2*i,j] = (x2-x1)/(2*e)
                A[2*i+1,j] = (y2-y1)/(2*e)
                
            i = i + 1
            
        return A
                
    #define rotation matrix for different order of rotations
    def rotmat_1(self, omega, phi, kappa):
        R = np.zeros((3, 3))
        
        R[0,0] = math.cos(phi)*math.cos(kappa)+math.sin(phi)*math.sin(omega)*math.sin(kappa)
        R[1,0] = math.sin(phi)*math.cos(kappa)-math.cos(phi)*math.sin(omega)*math.sin(kappa)
        R[2,0] = math.cos(omega)*math.sin(kappa)
        R[0,1] = -math.cos(phi)*math.sin(kappa)+math.sin(phi)*math.sin(omega)*math.cos(kappa)
        R[1,1] = -math.sin(phi)*math.sin(kappa)-math.cos(phi)*math.sin(omega)*math.cos(kappa)
        R[2,1] = math.cos(omega)*math.cos(kappa)
        R[0,2] = math.sin(phi)*math.cos(omega)
        R[1,2] = -math.cos(phi)*math.cos(omega)
        R[2,2] = -math.sin(omega)
        
        return R
    
    #define rotation matrix for different order of rotations
    def rotmat_2(self, omega, phi, kappa):
        R = np.zeros((3, 3))
        
        R[0,0] =  math.cos(phi)*math.cos(kappa)
        R[0,1] = -math.cos(phi)*math.sin(kappa)
        R[0,2] =  math.sin(phi)
        R[1,0] =  math.cos(omega)*math.sin(kappa)+math.sin(omega)*math.sin(phi)*math.cos(kappa)
        R[1,1] =  math.cos(omega)*math.cos(kappa)-math.sin(omega)*math.sin(phi)*math.sin(kappa)
        R[1,2] = -math.sin(omega)*math.cos(phi)
        R[2,0] =  math.sin(omega)*math.sin(kappa)-math.cos(omega)*math.sin(phi)*math.cos(kappa)
        R[2,1] =  math.sin(omega)*math.cos(kappa)+math.cos(omega)*math.sin(phi)*math.sin(kappa)
        R[2,2] =  math.cos(omega)*math.cos(phi)
        
        return R
    
    
    #general camera model (collinearity/telecentric equations)
    def model_resection(self, camera_interior, GCP, camera_exterior, rot_mat_dir_v1=True):
    #camera_exterior: coordiantes of projection centre  and angles of ration matrix (numpy array [X0, Y0, Z0, omega, phi, kappe])
    #GCP: ground control point coordinates (numpy array [X, Y, Z])
    #camera_interior: interior camera orientation (numpy array [ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0]), Brown (aicon) model
    #rot_mat_dir_v1: choose rotation matrix version
    
        ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0 = camera_interior
    
        ProjCentre = camera_exterior[0:3]
        RotMat = camera_exterior[3:6]
        
        if rot_mat_dir_v1:
            R = self.rotmat_2(RotMat[0], RotMat[1], RotMat[2])
            N = R[0,2]*(GCP[0]-ProjCentre[0]) + R[1,2]*(GCP[1]-ProjCentre[1]) + R[2,2]*(GCP[2]-ProjCentre[2])
        else: 
            R = self.rotmat_2(RotMat[0], RotMat[1], RotMat[2])
            N = -1
    
        kx = R[0,0]*(GCP[0]-ProjCentre[0]) + R[1,0]*(GCP[1]-ProjCentre[1]) + R[2,0]*(GCP[2]-ProjCentre[2])
        ky = R[0,1]*(GCP[0]-ProjCentre[0]) + R[1,1]*(GCP[1]-ProjCentre[1]) + R[2,1]*(GCP[2]-ProjCentre[2])
        
        x = -1*ck*(kx/N)
        y = -1*ck*(ky/N)
            
        r = np.sqrt(x*x+y*y)
        
        x = xh + x;
        x = x + x * (A1*(r**2-r0**2)+A2*(r**4-r0**4)+A3*(r**6-r0**6))
        x = x + B1*(r*r+2*x*x) + 2*B2*x*y
        x = x + C1*x + C2*y
        
        y = yh + y;
        y = y + y * (A1*(r**2-r0**2)+A2*(r**4-r0**4)+A3*(r**6-r0**6))
        y = y + B2*(r*r+2*y*y) + 2*B1*x*y
        y = y + 0
        
        
        return x, y
    
    
    #main function for spatial resection
    def resection(self, camera_interior, camera_exterior, ImgCoos_GCPCoos, e=0.0001, plot_results=False, dir_plot=None):
    #camera_exterior: estimate of exterior orientation and position (XYZOmegaPhiKappa)
    #camera_interior: interior camera orientation (numpy array [ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0]), Brown (aicon) model
    #ImgCoos_GCPCoos: assigned image coordinates and object coordinates of ground control points 
    #e: epsilon
        
        ImgCoos_GCPCoos_cl = TwoD_to_ThreeD.AssignedCoo()
        ImgCoos_GCPCoos = ImgCoos_GCPCoos_cl.mat_assignedCoo_all(ImgCoos_GCPCoos.x, ImgCoos_GCPCoos.y, ImgCoos_GCPCoos.X,
                                                                 ImgCoos_GCPCoos.Y, ImgCoos_GCPCoos.Z)
        
        '''iterative calculation of parameter values'''
        s0 = 0
        restart = False
        camera_exterior_ori = np.zeros((camera_exterior.shape[0],1))
        camera_exterior_ori[:] = camera_exterior[:]                               
        for iteration in range(200):
            
            #only if outlier in image measurement detected
            if restart:
                camera_exterior = np.zeros((camera_exterior_ori.shape[0],1))
                camera_exterior[:] = camera_exterior_ori[:]
                iteration = 0
                restart = False
            
            try:
            
                l = self.l_vector_resection(ImgCoos_GCPCoos, camera_interior, camera_exterior)
                A = self.A_mat_resection(ImgCoos_GCPCoos, camera_exterior, camera_interior, e)
            
                '''least squares adjustment'''
                N  = np.matrix(A.T) * np.matrix(A)
                L  = np.matrix(A.T) * np.matrix(l)
                Q  = np.matrix(np.linalg.inv(N))
                dx = Q * L  #N\L
                v  = np.matrix(A) * dx - np.matrix(l)
                s0 = np.sqrt((v.T * v) / (A.shape[0] - A.shape[1])) # sigma-0
                
    #             if iteration == 0:
    #                 print(v)
            
                ''''adds corrections to the values of unknowns'''
                SUM = 0
                for par_nbr in range(camera_exterior.shape[0]):
                    camera_exterior[par_nbr] = camera_exterior[par_nbr] + dx[par_nbr]
                    SUM = SUM + np.abs(dx[par_nbr])
                
                ''''stops the iteration if sum of additions is very small'''
                if (SUM < 0.00001):
                    break
                
                '''calculate std of corrections to check for outliers'''
                std_v = np.std(v)
                mean_v = np.mean(v)
                
                #remove point if larger 2*std
                for k in range(v.shape[0]):
                    if mean_v + 3 * std_v < v[k]:
                        #if k % 2 == 0:
                        print('outlier during resection detected: ', ImgCoos_GCPCoos[int(k/2)])
                        ImgCoos_GCPCoos = np.delete(ImgCoos_GCPCoos, (int(k/2)), axis=0)
                        restart = True
                        break
                if restart:
                    continue
                
            except Exception as error:
                print(error)
                return np.asarray([[-9999],[0]]), s0
            
    
    #     '''Output per iteration (check on convergence)'''
    #     print('Iteration ' + str(iteration))
    #     print('  Sigma-0: ' + str(s0) + ' mm')
    #     print('  Sum of additions: ' + str(SUM))
        
        if plot_results:
            '''Generation of vector field (residuals of image coordinates)'''
            #splits the x- and y-coordinates in two different vectors
            x = ImgCoos_GCPCoos[:,0]
            y = ImgCoos_GCPCoos[:,1]
            vx = np.zeros((v.shape[0],1))
            vy = np.zeros((v.shape[0],1))
            for i in range(ImgCoos_GCPCoos.shape[0]):
                vx[i] = v[i*2]
                vy[i] = v[i*2+1]
        
            #displays residuals in a seperate window
            set_markersize = 2    
            fontProperties_text = {'size' : 10, 
                                   'family' : 'serif'}
            matplotlib.rc('font', **fontProperties_text)    
            fig = plt.figure(frameon=False) 
            ax = plt.Axes(fig, [0., 0., 1., 1.])
        #    ax.set_axis_off()
            fig.add_axes(ax)     
            ax.plot(x, y, 'go')    
            
            a_scale = 1
            a_width = 0.05
            for xl, yl, v_x, v_y in zip(x, y, vx, vy):
                ax.arrow(xl, yl , v_x[0] * a_scale, v_y[0] * a_scale, head_width=a_width, 
                         head_length=a_width*1, fc='k', ec='k')  
        
            plt.show()
        
        '''Calculation of standard deviations of estimated parameters'''
        calibration_results = np.zeros((camera_exterior.shape[0],2))
        
        for j in range(camera_exterior.shape[0]):
            calibration_results[j,0] = camera_exterior[j]
            calibration_results[j,1] = s0 * np.sqrt(Q[j,j])
        
        #displays the estimated parameters incl. their standard deviation
        return calibration_results, s0  
