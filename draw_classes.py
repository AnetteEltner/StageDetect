from wx import App, ScreenDC    #to get monitor resolution
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, cv2


class Drawing:
    
    def __init__(self):
        pass
    
    
    '''----drawing results tools----'''
    
    '''get montior resolution in dpi'''
    def monitordpi(self):
        app = App(0)
        s = ScreenDC()
        monitordpi = s.GetPPI()[0]
        return monitordpi
    
    
    '''define different colors for specific number of values'''
    def color_spectrum(self, unique_vals, offset=35, color_type='spectral'):
    # unique_vals: type is list
    # offset to differentiate colors
    # color definitions
    # output is cmap color values for each data value
        cmap = plt.get_cmap(color_type)   #'binary'PiYG
        colors = []
        i = 0
        c = 0
        while i < len(unique_vals):
            colors.append(cmap(c))
            i=i+1
            c=c+offset
            
        return colors
    
    
    '''draw points on image'''
    def draw_points_onto_image(self, image, image_points, point_id, markSize=2, fontSize=8, switched=False):
    # draw image points into image and label the point id
    # image_points: array with 2 columns
    # point_id: list of point ids in same order as corresponding image_points file; if empty no points labeled
    # dpi from screen resolution
        dpi = self.monitordpi()
        
        set_markersize = markSize
        
        fontProperties_text = {'size' : fontSize, 
                               'family' : 'serif'}
        matplotlib.rc('font', **fontProperties_text)
        
        fig = plt.figure(frameon=False) #dpi of screen resolution
        fig.set_size_inches(image.shape[1]/float(dpi), image.shape[0]/float(dpi)) #dpi screen resolution!
        
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
         
        if switched:
            ax.plot([p[1] for p in image_points],
                    [p[0] for p in image_points],
                    marker='o', ms=set_markersize, color='green', markeredgecolor='green', markeredgewidth=1)
        else:
            ax.plot([p[0] for p in image_points],
                     [p[1] for p in image_points],
                     marker='o', ms=set_markersize, color='red', markeredgecolor='black', markeredgewidth=1)
                   
        #ax.plot(image_points[:,0], image_points[:,1], "r.", markersize=set_markersize, markeredgecolor='black')
        if len(point_id) > 1:
            if not switched:
                for label, xl, yl in zip(point_id, image_points[:,0], image_points[:,1]):
                    ax.annotate(str((label)), xy = (xl, yl), xytext=(xl+5, yl+1), color='blue', **fontProperties_text)
            else:
                for label, xl, yl in zip(point_id, image_points[:,1], image_points[:,0]):
                    ax.annotate(str((label)), xy = (xl, yl), xytext=(xl+5, yl+1), color='blue', **fontProperties_text)           #str(int(label)
    
        ax.imshow(image, cmap='gray', aspect='normal')
            
        return plt
    
    '''draw points on image'''
    def plot_pts(self, img, points, switchColRow=False, plt_title='', output_save=False, edgecolor='blue'):
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
            return plt
    
    '''draw Harris points on image'''   
    def plot_harris_points(self, image, filtered_coords, save=False, directory_img=None):
        """ Plots corners found in image. """
        
        plt.figure()
        plt.gray()
        plt.imshow(image)
        plt.plot([p[1] for p in filtered_coords],
                    [p[0] for p in filtered_coords],
                    marker='o', ms=2, color='none', markeredgecolor='blue', markeredgewidth=0.2)
        plt.axis('off')
        
        if save:
            plt.savefig(os.path.join(directory_img, 'harris.jpg'), dpi=600, pad_inches=0)
        else:
            plt.show()
        
    '''draw SIFT matches on images'''
    def plot_matches_SIFT(self, imagename1, imagename2, locs1, locs2, matchscores, show_below=True):
        '''Show a figure with lines joining the accepted matches
        input: im1, im2, (images as arrays), locs1, locs2 (feature locations),
        matchscores (as ouptut from 'match()'),
        show_below (if images should be shown below matches ). '''
        im1 = cv2.imread(imagename1)
        im2 = cv2.imread(imagename2)     
        
        im3 = self.appendimages(im1, im2)
    
        if show_below:
            #im3 = np.vstack((im3, im3))
            plt.imshow(im3)
            
            cols1 = im1.shape[1]
            for i,m in enumerate(matchscores):
                if m > 0:
                    plt.plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
                plt.axis('off')
    
    '''draw STAR matches on images'''
    def plot_matches(self, im1, im2, pts1, pts2, nbr_match_draw_set=0, save=False, directory_img=None):
        '''draw STAR matches
        im1, im2 location and name of images
        pts1, pts2 (numpy array): location of matched points in image
        nbr_match_draw: amount of matches to be displayed'''
        
        if nbr_match_draw_set == 0:
            nbr_match_draw = pts1.shape[0]    
        else:
            nbr_match_draw = nbr_match_draw_set
        
        img2_show = plt.imread(im2)
        if len(img2_show.shape) > 2:
            ymax2, xmax2, _ = img2_show.shape    #ymax2, xmax2, _ =
        else:
            ymax2, xmax2 = img2_show.shape
            
        img1_show = plt.imread(im1)
        if len(img1_show.shape) > 2:
            ymax1, xmax1, _ = img1_show.shape   
        else:
            ymax1, xmax1 = img1_show.shape
            
        if ymax1 > ymax2:
            ymax = ymax1
        else: 
            ymax = ymax2 
        
        fig = plt.figure(figsize=((xmax1+xmax2)/1000, (ymax)/1000))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plt.subplots_adjust(wspace=0, hspace=0)
        
        ax1.imshow(img2_show, aspect='auto') #clip white boarder
        ax2.imshow(img1_show, aspect='auto', cmap='Greys_r') #clip white boarder
        
        pts1_draw = np.asarray(pts1, dtype=np.float)
        pts2_draw = np.asarray(pts2, dtype=np.float)
    
        if len(pts1_draw.shape) == 3:   
            x1,y1 = pts1_draw[:,:,0:1].flatten(), pts1[:,:,1:2].flatten()
            x2,y2 = pts2_draw[:,:,0:1].flatten(), pts2[:,:,1:2].flatten()
        else:
            x1,y1 = pts1_draw[:,0:1].flatten(), pts1[:,1:2].flatten()
            x2,y2 = pts2_draw[:,0:1].flatten(), pts2[:,1:2].flatten()
           
        colors = self.color_spectrum(pts1_draw.tolist(), offset=1)
        
        print 'plotting matches'
        
        i = 0
        lines = []
        while i < nbr_match_draw:#pts1_draw.shape[0]:          
            transFigure = fig.transFigure.inverted()
             
            coord1 = transFigure.transform(ax1.transData.transform([x1[i],y1[i]]))
            coord2 = transFigure.transform(ax2.transData.transform([x2[i],y2[i]]))
           
            line = plt.matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                                            transform=fig.transFigure, color=colors[i])    #
        
            plt.setp(line, color=colors[i], linewidth=0.2)
            lines.append(line,)
            
            ax1.plot(x1[i], y1[i], marker='o', ms=1, color='none', markeredgecolor=colors[i], markeredgewidth=0.2) #        color=colors[i], markeredgecolor='none'
            ax2.plot(x2[i], y2[i], marker='o', ms=1, color='none', markeredgecolor=colors[i], markeredgewidth=0.2) 
    
            ax1.imshow(img2_show, aspect='auto')    #re-center image
            ax2.imshow(img1_show, aspect='auto', cmap='Greys_r')    #re-center image
            
            i = i+1    
            
        fig.lines = lines
        
        ax1.axis('off')
        ax2.axis('off')
        
        if save:
            plt.savefig(os.path.join(directory_img, 'matches.jpg'), dpi=600)
        else:
            plt.show()
        
        print 'plotting STAR matches done'
        
        return fig
    
    #draw image points on image     
    def plot_features(self, im, locs, circle=False):
        '''Show image with features. input: im (image as array), locs (row, col, scale, orientation of each feature).'''
        
        def draw_circle(c, r):
            t = np.arange(0,1.01,.01)*2*np.pi
            x = r*np.cos(t) + c[0]
            y = r*np.sin(t) + c[1]
            plt.plot(x,y,'b',linewidth=2)
            
        plt.imshow(im)
        if circle:
            for p in locs:
                draw_circle(p[:2],p[2])
        else:
            plt.plot(locs[:,0],locs[:,1],'ob')
            plt.axis('off')
    
    #help function to plot assigned SIFT features
    def appendimages(self, im1, im2):
        '''Return a new image that appends the two images side-by-side.'''    
        # select the image with the fewest rows and fill in enough empty rows
        rows1 = im1.shape[0]
        rows2 = im2.shape[0]
        
        if rows1 < rows2:
            im1 = np.vstack((im1, np.zeros((rows2-rows1, im1.shape[1], im1.shape[2]))))
        elif rows1 > rows2:
            im2 = np.vstack((im2, np.zeros((rows1-rows2, im2.shape[1],  im2.shape[2]))))
        # if none of these cases they are equal, no fillng needed.
    
        return np.concatenate((im1, im2), axis=1)