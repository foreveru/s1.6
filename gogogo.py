# -*- coding: utf-8 -*-
from __future__ import print_function

from time import time
from PIL  import Image, ImageOps
from io   import BytesIO

import os
import sys
import cv2
import math
import numpy as np
import base64
import operator
import logging
import collections
from imageprocess import ImageProcessor
from imageprocess import is_blue, is_green, is_red, is_black

#from logging.config import fileConfig
import logging

# fileConfig('logging_config.ini')
# logger = logging.getLogger('free')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler('output.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler(stream=None)
logger.addHandler(sh)
handler.setFormatter(formatter)
logger.addHandler(handler)

global g0area

DEBUG        = False
DEBUG_SIGN_D = False    # detect
DEBUG_SIGN_R = False    # recognize
DEBUG_OB     = False
DONTMOVE     = False

def sign_flatten_rgb(img):
    r, g, b = cv2.split(img)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))
    w_filter = ((r > 200) & (g > 200) & (b > 200))
    h_filter = ((r < 50) & (g < 50) & (b < 50))
    a_filter = ((r < 200) & (g < 100) & (b < 100))
    # r[y_filter], g[y_filter] = 255, 255
    # b[np.invert(y_filter)] = 0
    b[b_filter], b[np.invert(b_filter)] = 255, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0
    g[g_filter], g[np.invert(g_filter)] = 255, 0
    r[w_filter], g[w_filter], b[w_filter] = 255, 255, 255 
    r[h_filter], g[h_filter], b[h_filter] =255, 255, 255    
    flattened = cv2.merge((r, g, b))

    r, g, b = cv2.split(flattened)
    a_filter = ((r < 200) & (g < 100) & (b < 100))
    r[a_filter], g[a_filter], b[a_filter] =255, 255, 255 
    flattened = cv2.merge((r, g, b))

    r, g, b = cv2.split(flattened)
    d_filter = ((r < 100) & (g > 200) & (b < 100))
    r[d_filter], g[d_filter], b[d_filter] = 255, 255, 255 
    flattened = cv2.merge((r, g, b))

    return flattened

def od_flatten_rgb(img):
    r, g, b = cv2.split(img)
    a_filter = ((r < 40) & (g < 40) & (b < 40))
    r[a_filter], g[a_filter], b[a_filter] =255, 255, 255 
    flattened = cv2.merge((r, g, b))

    r, g, b = cv2.split(flattened)
    a_filter = ((r > 200) & (g > 200) & (b > 200))
    r[a_filter], g[a_filter], b[a_filter] =255, 255, 255 
    flattened = cv2.merge((r, g, b))
    
    return flattened
    
def preprocess(gray):

    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)

    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    element1  = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2  = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    dilation  = cv2.dilate(binary, element2, iterations = 1)
    erosion   = cv2.erode(dilation, element1, iterations = 1)
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)

    return dilation2
    
def sign_find_region(img):

    region = []
    area = 0

    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img2, contours, -1, (0,255,0), 3)
    max_area   = 0
    index_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if ( area > max_area ):
            index_area = i
            max_area = area

    if ( len(contours) > 0 and max_area > 700 ) :
        cnt = contours[index_area]

        area = cv2.contourArea(cnt) 

        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
 
        rect = cv2.minAreaRect(cnt)

        box = cv2.boxPoints(rect)
        box = np.int0(box)
 
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        if ( DEBUG == True and DEBUG_SIGN_D == True):
            print("ROI: area=%d, width=%d, height=%d" % (area, width, height))

        if (height > width * 1.2 or width > height * 5 or height < 10 ):
            return [], 0

        region.append(box)
 
    return region, area

def od_find_region(img):

    region = []

    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt) 

        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
 
        rect = cv2.minAreaRect(cnt)

        box = cv2.boxPoints(rect)
        box = np.int0(box)
 
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
 
        if(height > width * 1.2):
            continue
 
        region.append(box)
 
    return region
    

def xorcmp(s1, s2):
    a1 = cv2.resize(s1,(32,32),interpolation=cv2.INTER_CUBIC)
    a2 = cv2.resize(s2,(32,32),interpolation=cv2.INTER_CUBIC)
    return cv2.bitwise_xor(a1, a2)

def logit(msg):
    print("%s" % msg)

class PID:
    def __init__(self, Kp, Ki, Kd, max_integral, min_interval=0.001, set_point=0.0, last_time=None):
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._min_interval = min_interval
        self._max_integral = max_integral

        self._set_point = set_point
        self._last_time = last_time if last_time is not None else time()
        self._p_value = 0.0
        self._i_value = 0.0
        self._d_value = 0.0

        self._delta_time = 0.0
        self._delta_error = 0.0
        self._last_error = 0.0
        self._output = 0.0

    def update(self, cur_value, cur_time = None):
        if cur_time is None:
            cur_time = time()

        error = self._set_point - cur_value
        d_time = cur_time - self._last_time
        d_error = error - self._last_error

        if d_time >= self._min_interval:
            self._p_value = error
            self._i_value = min(max(error * d_time, -self._max_integral), self._max_integral)
            self._d_value = d_error / d_time if d_time > 0 else 0.0
            self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd

            self._delta_time = d_time
            self._delta_error = d_error
            self._last_time = cur_time
            self._last_error = error

        return self._output

    def reset(self, last_time = None, set_point = 0.0):
        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._delta_time       = 0.0
        self._delta_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0

    def assign_set_point(self, set_point):
        self._set_point = set_point

    def get_set_point(self):
        return self._set_point

    def get_p_value(self):
        return self._p_value

    def get_i_value(self):
        return self._i_value

    def get_d_value(self):
        return self._d_value

    def get_delta_time(self):
        return self._delta_time

    def get_delta_error(self):
        return self._delta_error

    def get_last_error(self):
        return self._last_error

    def get_last_time(self):
        return self._last_time

    def get_output(self):
        return self._output




class Car(object):
    MAX_STEERING_ANGLE = 40.0
    ############# Add PID control 09/30 ###
    THROTTLE_PID_Kp = 0.13  #0.02
    THROTTLE_PID_Ki = 0.005
    THROTTLE_PID_Kd =  0.01   # 0.02
    THROTTLE_PID_max_integral = 0.5
    MAX_STEERING_HISTORY = 3
    MAX_THROTTLE_HISTORY = 3
    DEFAULT_SPEED = 2.2
    ##########################################

    def __init__(self, control_function):
        self._control_function = control_function
        self.servo_err_history = collections.deque([0], 30)
        self.motor_err_history = collections.deque([0, 0, 0, 0, 0, 0], 90)
        self.speed_sum = 0
        self.last_time = 0
        self.steering_angle = 0
        self.throttle = 0
        self.car_status = 'INITIAL'
        self.track = 3
        self.LOW_SPEED = 1.90
        self.last_throttle = 0
        self.across_flag = 0
        
        self.delay_return_mid = 0
        self.last_track_count = 0
        self.ct = time()
        self.frame_counter = 0
        self.main_road = True
        self.branch_road = False
        self.vision_history = []
        self.vision_history_size = []
        self.time_counter = 0

        #### add throttle PID control 0930 ###########################
        self.throttle_pid = PID(Kp=self.THROTTLE_PID_Kp, Ki=self.THROTTLE_PID_Ki, Kd=self.THROTTLE_PID_Kd,
                                max_integral=self.THROTTLE_PID_max_integral)
        self.throttle_pid.assign_set_point(self.DEFAULT_SPEED)
        self.throttle_history = []
        #### added by Elsie  #########################################
        self.first_frame = True # 是否是第一帧
        self.total_frame = 0 # 当前总帧数

        # if crash, then back
        self.total_back_frames = 15
        self.curr_back_frame = 0

        # after back, then make a turn to go on driving
        self.total_turns = 5
        self.curr_turn = 0

        self.mark = 1
        ##############################################################


        # "|-->": fork in the road ahead, the track will fork in two, follow the right lane.
        # "<--|": fork int the road ahead, the track will fork in two, follow the left lane.
        # "V--+": U turn to the left ahead.
        # "+--V": U turn to the right ahead.
        sign_filename_list =["sign-02-custom.png", "sign-03-custom.png", "sign-04-custom.png", "sign-05-custom.png", "sign-07-custom.png", "sign-08-custom.png", "sign-09-custom.png", "sign-10-custom.png"]
        self.sign_sym_list =['X-->', "<---", "<--X", "--->", "|-->", "<--|", "V--+", "+--V"]        
        self.sign_img_list  = []
    
        for filename in sign_filename_list :
            img = cv2.imread(filename, 0)
            self.sign_img_list.append(img)
        
        for i in range(0, 30):
            self.servo_err_history.appendleft(0)

        self.init_cnt = 10

    def show_ROI(self, title, img, region):
    
        for box in region:

            if ( box[2][1] > box[0][1] ):
                a = box[0][1]
                b = box[2][1]
            else:
                b = box[0][1]
                a = box[2][1]
            if ( box[0][0] > box[2][0] ):
                c = box[2][0]
                d = box[0][0]
            else:
                d = box[2][0]
                c = box[0][0]

            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            cv2.imshow(title, img)

    def recognize_sign(self, img):
        
        h = img.shape[0]
        w = img.shape[1]
        
        if ( w < 20 ):
            return ""
            
        # only compare with left_turn and left_right
        if ( (float(w) / float(h)) > 3 ):
            xor_left  = xorcmp(img, self.sign_img_list[1])
            xor_right = xorcmp(img, self.sign_img_list[3])
            if (xor_left.mean() > xor_right.mean()):
                return self.sign_sym_list[3]
            
            return self.sign_sym_list[1]
        
        # compare with all samples
        xor_value = np.arange(8)
        xor_value = np.float32(xor_value)
        i = 0
        for sample in self.sign_img_list:
            xorimg = xorcmp(img, sample)
            xor_value[i] = xorimg.mean()
            if (DEBUG == True and DEBUG_SIGN_R == True):
                cv2.imshow("xor "+ self.sign_sym_list[i], xorimg)
            i = i + 1
        
        xmin, xmax = xor_value.min(), xor_value.max()
        xor_value = (xor_value - xmin) / ( xmax - xmin )
        xmin = xor_value.min()
        i = 0
        for v in xor_value:
            if ( v == xmin ):
                break
            i = i + 1

        if (DEBUG == True):
            print(xor_value, xmin, self.sign_sym_list[i])

        #logger.info('%f %s', xmin, self.sign_sym_list[i])
        
        return self.sign_sym_list[i]
        
    def obstacle_detect(self, img):
    
        showimg = img.copy()
        
        img = od_flatten_rgb(img)
        if (DEBUG == True and DEBUG_OB == True):  
            cv2.imshow("OB flatten", img)
            
        info =  img.shape
        h = info[0]
        w = info[1]
        d = info[2]
        
        v = np.var(img, axis=2)

        v[ v < 60] = 0
        v[ v > 60 ] = 255

        bgr = np.split(img, 3, axis=2)

        b = bgr[0].reshape(h,w)
        g = bgr[1].reshape(h,w)
        r = bgr[2].reshape(h,w)

        v = np.uint8(v)

        b = np.bitwise_or(b, v)
        g = np.bitwise_or(g, v)
        r = np.bitwise_or(r, v)

        final = np.concatenate( (b.reshape(h,w,1), g.reshape(h,w,1), r.reshape(h,w,1)), axis=2)

        denoiseimg = cv2.medianBlur(final,5)
        
        if (DEBUG == True and DEBUG_OB == True):  
            cv2.imshow("OB variance", denoiseimg)
            
        gray = cv2.cvtColor(denoiseimg, cv2.COLOR_BGR2GRAY)

        dilation = preprocess(gray)
        if (DEBUG == True and DEBUG_OB == True):  
            cv2.imshow("OB preprocess", denoiseimg)
            
        region = od_find_region(dilation)
            
        if (DEBUG == True):
            self.show_ROI("OD ROI", showimg, region)
        
        return img
    

    def sign_detect(self, img):
        
        showimg = img.copy()
        
        img = sign_flatten_rgb(img) # 找到红色箭头标志，其余部分变白色
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 红色箭头灰度图
        if (DEBUG == True and DEBUG_SIGN_D == True):  
            cv2.imshow("Gray", gray)

        dilation = preprocess(gray)
        if (DEBUG == True and DEBUG_SIGN_D == True):          
            cv2.imshow("Preprocess", dilation)
            
        region, area = sign_find_region(dilation)
        if (DEBUG == True):
            self.show_ROI("Sign ROI", showimg, region)
        
        vision_history_len = len(self.vision_history_size)
        if ( area == 0 and vision_history_len > 0 ):
            g0area.append(0)

        # during 10*100ms, car can not see anything, clean vision history
        if (vision_history_len > 0 and len(g0area) >  10 ):
            del self.vision_history_size[:]
            del self.vision_history[:]
            del g0area[:]
            if (DEBUG == True):
                print("clean vision history. %d" % len(g0area))
            return ""

        # Only condition is true, Do recognize. 
        if ( vision_history_len > 4 and len(g0area) >  3 ):
            try:
                max_index, max_area = max(enumerate(self.vision_history_size), key=operator.itemgetter(1))
            except ValueError as e:
                del self.vision_history_size[:]
                del self.vision_history[:]
                del g0area[:]

                return ""

            if (DEBUG == True):
                cv2.imshow("aaaa", self.vision_history[max_index])
                print(self.vision_history_size)
                print("max vision size = %d, vision history len = %d, See nothing = %d" % (self.vision_history_size[max_index], vision_history_len, len(g0area)))

            cropimg = self.vision_history[max_index]

            # remove border
            arr = Image.fromarray(cropimg)
            pilimg = arr.convert('L')
            invert_img = ImageOps.invert(pilimg)
            box =invert_img.getbbox()

            flagimg = cropimg[ box[1]:box[3], box[0]:box[2] ]

            del self.vision_history_size[:]
            del self.vision_history[:]
            del g0area[:]

            if (DEBUG == True and DEBUG_SIGN_D == True):  
                cv2.imshow("Binary Image", flagimg)

            return self.recognize_sign(flagimg)
            
        for box in region:

            if ( box[2][1] > box[0][1] ):
                a = box[0][1]
                b = box[2][1]
            else:
                b = box[0][1]
                a = box[2][1]
            if ( box[0][0] > box[2][0] ):
                c = box[2][0]
                d = box[0][0]
            else:
                d = box[2][0]
                c = box[0][0]

            cropimg = gray[ a:b, c:d ]

            # cropimg = cv2.fastNlMeansDenoising(cropimg,None,10,10,5,5)
            cropimg = cv2.medianBlur(cropimg,5)
            self.vision_history_size.append( area )
            self.vision_history.append( cropimg )

        return ""

    def adjust_track(self, track_count, last_track_count):
        if (track_count > 4):
            if (self.across_flag < 0):  # to left
                if ( self.track[8] > 3):
                    self.track[8] = self.track[8] - 2
                elif ( self.track[8] > 2):
                    self.track[8] = self.track[8] - 1
            elif  (self.across_flag > 0):  # to right
                if ( self.track[8] < 4):
                    self.track[8] = self.track[8] + 2
                elif ( self.track[8] < 5):
                    self.track[8] = self.track[8] + 1

            if (self.across_flag !=0):
                self.delay_return_mid = 50
                self.across_flag = 0
                if (DEBUG == True):
                    print("Due to road flag, delay_return_mid = 50")
                return 

        if (self.delay_return_mid > 0):
            self.delay_return_mid  = self.delay_return_mid - 1
            return 

        if (track_count < 6 and self.delay_return_mid == 0):
            self.delay_return_mid == 10
            if (DEBUG == True and self.branch_road == False):
                print("enter branch, delay_return_mid = 10")
                self.branch_road = True
                self.main_road = False

        if (last_track_count < 6 and track_count == 6):
            self.delay_return_mid = 10
            if (DEBUG == True):
                print("see main road, delay_return_mid = 10")

        if (track_count == 6 and self.delay_return_mid == 0 ):
            if (DEBUG == True and self.main_road == False):
                print("enter main road")
                self.main_road = True
                self.branch_road = False

            if (self.track[8] != 3 ):
                if (self.track[8] > 3 ):
                    self.track[8] = self.track[8] - 1
                elif (self.track[8] < 3 ):
                    self.track[8] = self.track[8] + 1

                delay_return_mid = 10
                if (DEBUG == True):
                    print("change track in main road, delay_return_mid = 10")

        return

    def check_need_back_when_crash(self, srcimg, speed, curr_frame):
        shape = srcimg.shape
        img_hsv = cv2.cvtColor(srcimg, cv2.COLOR_RGB2HSV)
        #ImageProcessor.show_image(img_hsv, 'hsv_img')
        mask_black = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 46])) / 255
        mask_yellow = cv2.inRange(img_hsv, np.array([27, 42, 165]), np.array([34, 255, 255])) / 255
        black_sum = np.sum(mask_black)
        yellow_sum = np.sum(mask_yellow)
        ratio_b = black_sum / shape[0] / shape[1]
        ratio_y = yellow_sum / shape[0] / shape[1]
        logger.info("ratio_black=%s, ratio_yellow=%s" % (ratio_b, ratio_y))
        return [ratio_y, ratio_b]
        # if ratio_b > 0.5 or ratio_y > 0.5:
        #     return True   # the car has crashed
        # else:
        #     return False


    def on_dashboard(self, dashboard):
        self.total_frame = self.total_frame + 1
        logger.info("----------------------- frame: %s --------------------------" % self.total_frame)
        if ((time() - self.ct) < 1):
            self.frame_counter = self.frame_counter + 1
        else:
            if (self.time_counter % 20 == 0):
                logger.info('Frame rate = %d/s', self.frame_counter)
            self.ct = time()
            self.frame_counter = 0
            self.time_counter = self.time_counter + 1     
            
        #normalize the units of all parameters
        lap = float(dashboard["lap"])
        last_steering_angle = np.pi / 2 - float(dashboard["steering_angle"]) / 180.0 * np.pi
        throttle = float(dashboard["throttle"])
        brake = float(dashboard["brakes"])
        speed               = float(dashboard["speed"])
        imgsrc = Image.open(BytesIO(base64.b64decode(dashboard["image"])))
        oriimg              = np.asarray(imgsrc)  # (240,320,3)

        this_time = dashboard['time']
        #print(this_time)
        if this_time == self.last_time:
            #print(this_time)
            self.control(self.steering_angle, self.throttle)
            return
        self.last_time = this_time

        #################### Detection Start -------------        
        if (DEBUG == True):
            ImageProcessor.show_image(oriimg, "vision")  # 显示原RGB图像
        
        imgupper  = oriimg[0:90, 0:319].copy()  # 原图像上面部分(可能会有方向sign)
        imgbottom = oriimg[110:200, 0:319].copy()  # 原图像下面部分
        
        detect_start = time()
        sign  = self.sign_detect(imgupper)
        # imgbottom = self.obstacle_detect(imgbottom)       

        detect_end = time()
        
        if (sign == '<--X' or sign == '<--|'):
            self.across_flag = -2
        elif (sign == 'X-->' or sign == '|-->'):
            self.across_flag = 2

        # print("sign_detect + obstacle_detect cost = %fms" % ((detect_end - detect_start)*1000))
        #################### Detection End ...........
        
        if (DONTMOVE == True):
            self.control(0, 0)
            return 
        
        # preprocess of raw image
        nocropimg = ImageProcessor.preprocess(oriimg)
        # apply filter
        filterimg = ImageProcessor.image_filter(nocropimg)
        # perspective transformation
        img = self.perspective(filterimg)
        #ImageProcessor.show_image(img, "nocropimg")
        #ImageProcessor.save_image('log', img)
        #
        # car state machine
        # 
        if self.car_status == 'INITIAL':
            self.track = ImageProcessor.which_color_and_track(nocropimg)  # 当前图像中从左到右赛道的颜色，以及当前小车在哪个赛道
            self.car_status = 'RUNNING'
            self.track = list(self.track)
            # self.track[8] = 2

        track_count = self.how_many_tracks(filterimg)  # 当前图像中有几条赛道

        self.adjust_track(track_count, self.last_track_count)  # 根据赛道条数，当前所在的赛道位置，当前traffic sign等信息调整小车赛道位置
        self.last_track_count = track_count

        # normal
        mid, lost1 = self.determine_middle(img, row_default=20, color=self.track) # mid:小车在当前赛道纵坐标为row_default处的中心位置，
        mid2, lost2 = self.determine_middle(img, row_default=5, color=self.track)
        #logger.info("self.frame_counter = %s, mid1 = %s, mid2 = %s" % (self.frame_counter, mid, mid2))
        print("self.frame_counter = %s,mid1=%s, mid2=%s" %(self.frame_counter, mid, mid2))

        ########     Add by Elsie      #################################################################################
        # TODO 判断小车是否撞墙，是否需要倒车
        [ratio_y, ratio_b]= self.check_need_back_when_crash(oriimg, speed, self.total_frame)
        need_back = False
        if ratio_b > 0.4 or ratio_y > 0.4:
            need_back = True   # the car has crashed
        logger.info("Current Frame: %s: need_back=%s, speed=%s" %(self.total_frame, need_back, speed))
        if need_back:
            self.mark = 0
            if self.curr_back_frame < self.total_back_frames:
                self.curr_back_frame = self.curr_back_frame + 1
                self.control(0.0, -1.0)  # 倒车
                logger.info("in if-1")
                return
            else:
                self.curr_back_frame = 0
                self.control(40, 1.0)
                return
                # if self.curr_turn < self.total_turns:
                #     self.curr_turn = self.curr_turn + 1
                #     self.control(40, 1.0)
                #     logger.info("in if-2")
                #     return
                # else:
                #     self.mark = 1
                #     self.curr_back_frame = 0
                #     self.curr_turn = 0
                #     self.control(42, 0.1)  # 右转
                #     logger.info("in if-3")
                #     return
        logger.info("self.curr_back_frame=%s, self.curr_turn=%s" % (self.curr_back_frame, self.curr_turn))
        if self.first_frame:
            self.first_frame = False

        ##==============================================================================================================
        # calculate set speed
        set_speed = self.cal_speed(lost2)
        set_speed = 2 if set_speed>2 else set_speed



        if lost1 == False:
            self.steering_angle = self.servo_control(mid, mid2, lost1, lost2)
            # self.throttle = self.motor_control(mid, speed, set_default=set_speed) # original
            self.throttle = self.throttle_pid.update(speed)
        else:
            # if lost line, slow down and keep original direction
            self.throttle = self.motor_control(mid, speed, set_default=1)
            if self.steering_angle > 0:
                self.steering_angle += 0.8
            if self.steering_angle < -0:
                self.steering_angle -= 0.8
            #ImageProcessor.save_image('log', oriimg)
        #if (DEBUG == True):
        #    print (mid, mid2, lost1, lost2, self.steering_angle, self.track, self.car_status, set_speed, ' trackcount:', track_count)


        ####### Add by Elsie 0930 ############
        self.throttle_history.append(self.throttle)
        self.throttle_history = self.throttle_history[-self.MAX_THROTTLE_HISTORY:]
        self.throttle = sum(self.throttle_history[-15:]) / self.MAX_THROTTLE_HISTORY
        ######################################
        self.control(self.steering_angle, self.throttle)

        # 保存当前帧
        text0 = "Frame: %s, speed=%s, " % (self.total_frame, speed)
        text5 = "throttle=%s" % throttle
        text1= " ratio_yellow=%s, " % ratio_y
        text6 = "ratio_black=%s" %  ratio_b
        text2 = "need_back=%s,curr_back_frame=%s" % (need_back, self.curr_back_frame)
        text3 = "lap=%s, time=%s" % (lap, this_time)
        text4 = "brakes=%s, steering_angle=%s" % (brake, last_steering_angle )
        text7 = "current_track_count=%s" % track_count
        text8 = "self.track = %s" %self.track
        oriimg = cv2.cvtColor(oriimg, cv2.COLOR_BGR2RGB)
        cv2.putText(oriimg, text0, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text5, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text1, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text6, (10, 70), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text2, (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text3, (10, 110), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text4, (10, 130), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text7, (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        cv2.putText(oriimg, text8, (10, 170), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        filename = "%s.jpg" % self.total_frame
        cv2.imwrite(os.path.join('IMG', filename), oriimg, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


    def control(self, steering_angle, throttle):
        self._control_function(steering_angle, throttle)

    def servo_control(self, mid1, mid2, lost1, lost2):
        err = mid1 - 160
        self.servo_err_history.appendleft(err)
        delta_err = err - self.servo_err_history[5]

        if abs(err) < 40:
            Kp = 0.09  # 在合理的范围内Kp越大控制的效果越好（越快速的回到参考线附近）
            Kd = 0.02  # 0.02  # 增大D系数会增大无人车快速向参考线方向的运动的“抵抗力”从而使得向参考线方向的运动变得更加平滑
        elif abs(err) < 15:
            Kp = 0.075
            Kd = 0.02   # 0.02
        else:
            Kp = 0.09
            Kd = 0.02   # 0.02
        #Kp = 0.14
        #Kd = 0.03
        angle = Kp * err + Kd * delta_err
        #print(err, angle, delta_err, Kp * err, Kd * delta_err)
        # add filter, because steer is too sensitive
        return int(angle*3)/3.0

    def cal_speed(self, lost2):
        # in straight line a long time
        cnt = 0
        for err in self.servo_err_history:
            if abs(err) > 6:
                break
            cnt += 1
        #print(cnt)
        if cnt < 2:
            return self.LOW_SPEED
        if cnt < 4:
            return self.LOW_SPEED + 0.13  #0.05
        if cnt < 6:
            return self.LOW_SPEED + (0 if lost2 else 0.16)  # 0.1
        if cnt < 8:
            return self.LOW_SPEED + (0 if lost2 else 0.18)  # 0.15
        else:
            return self.LOW_SPEED + (0 if lost2 else 0.2)   # 0.2

    def motor_control(self, middle, speed, set_default=0.5):
        # default value
        # Kp = 0.12
        # Ki = 0.001
        Kp = 0.15
        Ki = 0.001

        set_speed = set_default
        err = set_speed - speed
        delta_err = err - self.motor_err_history[0]
        self.motor_err_history.appendleft(err)
        #self.speed_sum = sum(self.motor_err_history)

        throttle = self.last_throttle + delta_err * Kp + Ki * err
        if throttle < -0.4:
            throttle = -0.4
        if throttle > 1:
            throttle = 1
        self.last_throttle = throttle
        #print (err, self.speed_sum, err * Kp, Ki * self.speed_sum)
        return throttle

    # color is not used in this version
    def find_track(self, img, irow, color=2):
        left = -1
        right = 320

        track = self.track
        which_track = track[8]
        color = track[which_track]
        l_color = track[which_track-1]
        r_color = track[which_track+1]
        func_mapping = {
            0 : is_red,
            1 : is_green,
            2 : is_blue
        }
        is_color = func_mapping[color]
        is_l_color = func_mapping[l_color]
        is_r_color = func_mapping[r_color]

        # count from left to right
        # in right-center or left blue
        if which_track == 4 or which_track == 2:
            for i in range(0, 320):
                if is_color(img[irow, i]):
                    left = i
                    for j in range(i if i+10> 320 else i+10, 320):
                        if not is_color(img[irow, j]):
                            right = j
                            break
                    break
        # count from right to left
        # in left-center or right blue
        if which_track == 3 or which_track == 5:
            for i in range(319, -1, -1):
                if is_color(img[irow, i]):
                    right = i
                    for j in range(i if i-10 < 0 else i-10, -1, -1):
                        if not is_color(img[irow, j]):
                            left = j
                            break
                    break

        l_valid = False
        r_valid = False
        for i in range(left-1, -1 , -1):
            if is_black(img[irow, i]):
                continue
            if is_l_color(img[irow, i]):
                # should not too far
                if abs(left - i) < 30:
                    l_valid = True
            #else:
                break
        for i in range(right+1, 320):
            if is_black(img[irow, i]):
                continue
            if is_r_color(img[irow, i]):
                # should not too far
                if abs(right - i) < 30:
                    r_valid = True
                #print (1, i-right)
            #else:
                break
        if l_valid or r_valid:
            return left, right
        else:
            return -1, 320

        
    def perspective(self, img):
        cropimg = img[130:240, :]

        src = np.array([
            [0, 0],
            [320 , 0],
            [0, 110 ],
            [320 , 110]
            ], dtype = "float32")

        dst = np.array([
            [0, 0],
            [320, 0],
            [320 * 0.35, 110],
            [320 * 0.65, 110]], dtype = "float32") 

        M = cv2.getPerspectiveTransform(src, dst)
        wrapped = cv2.warpPerspective(cropimg, M, (320, 110))
        return  wrapped

    def determine_middle(self, img, row_default=18, color=2): # middle: 小车所在赛道的中心位置，
        #ImageProcessor.show_image(img, "source")

        left, right = self.find_track(img, row_default, color=color)

        if right < 100:
            middle = 50
        elif left > 220:
            middle = 270
        else:
            middle = (right+left)/2
        
        lost = False
        if right == 320 or left == -1:
            lost = True
        return middle, lost

    # how many tracks can we see in the camera
    def how_many_tracks(self, img):
        which_track = self.track[8]

        func_mapping = {
            0 : is_red,
            1 : is_green,
            2 : is_blue
        }

        if which_track == 4 or which_track == 3:
            return 6
        if which_track == 2:
            # in left blue, check right border if has track 4 color 
            is_color = func_mapping[self.track[4]]
            for i in range(239, 70, -1):
                # means we can see track 4's color
                if is_color(img[i, 290]):
                    return 6
            return 3
        if which_track == 5:
            # in right blue, check left border if has track 3 color
            is_color = func_mapping[self.track[3]]
            for i in range(239, 70, -1):
                if is_color(img[i, 30]):
                    return 6
            return 3
        
        return 3

if __name__ == "__main__":
    import shutil
    import argparse
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask
    g0area     = []
    sio = socketio.Server()
    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)

    car = Car(control_function = send_control)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)