
import numpy as np

class Detector():
    def __init__(self):
        self.prev_poly = [] 
        self.detect = True
        self.left_line_indxs =[]
        self.right_line_indxs =[]
        self.num_frames = 15
        self.prev_polynomials=[]
        self.best_polynomial=[]

    def get_lane_base(self,img,kernel_w):
        width=img.shape[1]
        height=img.shape[0]

        oneD_kernel = np.ones(kernel_w)
        # print(oneD_kernel)
        
        #trim {lower quarter horizontal} slice of img to detect first centre of Lane lines
        lower_left = img [int((1/2)*height):,:int(width/2)]
        lower_right = img [int((1/2)*height):,int(width/2):]
        l_sum_vert = np.sum(lower_left,axis=0)
        r_sum_vert = np.sum(lower_right,axis=0)
        # print(l_sum_vert.shape)
        
        left_centre = np.argmax (l_sum_vert)
        right_centre = np.argmax (r_sum_vert) + int(width/2)


        return left_centre,right_centre

    

    def detect_lanes(self,b_img,win_width=100):
        
        out = np.dstack((np.zeros_like(b_img),np.zeros_like(b_img),np.zeros_like(b_img)))
        width=b_img.shape[1]
        height=b_img.shape[0]
        n_windows = 12
        margin = win_width #margin left and right to search within it 
        window_height = (height / n_windows)
        ########    minlanepts missing
        whitepxls = b_img.nonzero()         #return tuple of 2 array contains x and y indxs to pixels with nonzero value 
        whitepxls_x = whitepxls[1]          # X indxs of nonzeroes  
        whitepxls_y = whitepxls[0]          # Y indxs of nonzeroes 

        if self.detect:

            left_line_indxs = [] # to stores indecies of points of line
            right_line_indxs = []
            left_x,right_x = self.get_lane_base(b_img,kernel_w=win_width)
            for window in range(n_windows,0,-1):
                
                #limits of window on y dir
                lower_y_window=  int((window-1)*window_height)
                upper_y_window= int((window)*window_height)
                # print ('leftx,rightx = ',left_x,right_x)
                # print ('lowery,uppery = ',lower_y_window,upper_y_window)
                #limits for search in x dir for both left and right lines
                #left limits
                lower_left_x = int( max(left_x-margin,0) )
                upper_left_x = int( min(left_x+margin,width) )
                #right limits
                lower_right_x = int( max(right_x-margin,0) )
                upper_right_x = int( min(right_x+margin,width) )

                #REMINDER:::visualize windows
                # cv2.rectangle(windows_img, (lower_left_x, lower_y_window), (upper_left_x, upper_y_window),(0, 255, 0), 2)
                # cv2.rectangle(windows_img, (lower_right_x, lower_y_window), (upper_right_x, upper_y_window),(0, 255, 0), 2)
                #select white pixels in each window (right and left)

                pixel_indxs_in_y= (whitepxls_y >= lower_y_window) & (whitepxls_y <= upper_y_window)
                left_pxls_indxs = (whitepxls_x >= lower_left_x) & (whitepxls_x <= upper_left_x)& pixel_indxs_in_y
                right_pxls_indxs = (whitepxls_x >= lower_right_x) & (whitepxls_x <= upper_right_x)& pixel_indxs_in_y
                left_pxls_indxs = left_pxls_indxs.nonzero()[0] 
                right_pxls_indxs = right_pxls_indxs.nonzero()[0]

                #if number of white pixels in window satisfy minimum number of pixels (here it's 50 pixel)         
                #it take the mean index in the window as the next centre
                if (len(left_pxls_indxs)>50):
                    whitexs = whitepxls_x [left_pxls_indxs]
                    left_x = int( np.mean(whitexs) )
                    # print('new left x = ',left_x)
                if (len(right_pxls_indxs)>50):
                    whitexs = whitepxls_x [right_pxls_indxs]
                    right_x = int( np.mean(whitexs) )
                    # print('new right x = ',right_x)

                left_line_indxs.append(left_pxls_indxs)
                right_line_indxs.append(right_pxls_indxs)
            
            #stacking each (left and right) indxs in one vector 
            self.left_line_indxs= np.hstack(left_line_indxs)
            self.right_line_indxs= np.hstack(right_line_indxs)
        else:
            left_poly = self.best_polynomial[0]
            right_poly = self.best_polynomial[1]
            self.left_line_indxs = ((whitepxls_x > (left_poly[0]*(whitepxls_y**2) + left_poly[1]*whitepxls_y + 
                                    left_poly[2] - margin)) & (whitepxls_x < (left_poly[0]*(whitepxls_y**2) + 
                                    left_poly[1]*whitepxls_y + left_poly[2] + margin))) 
            self.right_line_indxs = ((whitepxls_x > (right_poly[0]*(whitepxls_y**2) + right_poly[1]*whitepxls_y + 
                                    right_poly[2] - margin)) & (whitepxls_x < (right_poly[0]*(whitepxls_y**2) + 
                                    right_poly[1]*whitepxls_y + right_poly[2] + margin))) 
        #-------------------------------------------------------------------------------------------                            
        #get allpixels indexes (x,y)
        left_line_X =  whitepxls_x[self.left_line_indxs]
        left_line_Y =  whitepxls_y[self.left_line_indxs]
        right_line_X =  whitepxls_x[self.right_line_indxs]
        right_line_Y =  whitepxls_y[self.right_line_indxs] ###IFTAR TIME


        #REMNIDER :: visualize detected lines
        # out[left_line_Y,left_line_X,:]= [255,0,0]
        # out[right_line_Y,right_line_X,:]= [0,255,255]

        if left_line_Y.shape[0] >= 400 and right_line_Y.shape[0] >= 400 and left_line_X.shape[0] >= 400 and right_line_X.shape[0] >= 400:
            self.detect =False

            ### get best polynomial ##############################
            self.left_polynomial = np.polyfit(left_line_Y, left_line_X, 2)
            self.right_polynomial = np.polyfit(right_line_Y, right_line_X, 2)
            if (len(self.prev_polynomials)>=self.num_frames):
                self.prev_polynomials.pop(0)
            self.prev_polynomials.append([self.left_polynomial,self.right_polynomial])
            self.best_polynomial = np.mean(self.prev_polynomials,axis=0)
        else:
            self.detect = True  
            
        return self.best_polynomial