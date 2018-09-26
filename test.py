from imageprocess import *

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from io import BytesIO
import base64
import numpy as np
from gogogo import Car
from gogogo import *
import cv2

DEBUG = True
DEBUG_SIGN_D = True
def sign_flattern_rgb(img):
    r, g, b = cv2.split(img)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    print(r_filter)
    print(r_filter.shape)
    ImageProcessor.show_image(imgupper, "r_filter")
    print("--------------------------")
    print(g_filter)
    print(g_filter.shape)
    ImageProcessor.show_image(imgupper, "g_filter")
    print("--------------------------")
    print(b_filter)
    print(b_filter.shape)
    ImageProcessor.show_image(imgupper, "b_filter")
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
    r[h_filter], g[h_filter], b[h_filter] = 255, 255, 255
    flattened = cv2.merge((r, g, b))

    r, g, b = cv2.split(flattened)
    a_filter = ((r < 200) & (g < 100) & (b < 100))
    r[a_filter], g[a_filter], b[a_filter] = 255, 255, 255
    flattened = cv2.merge((r, g, b))

    r, g, b = cv2.split(flattened)
    d_filter = ((r < 100) & (g > 200) & (b < 100))
    r[d_filter], g[d_filter], b[d_filter] = 255, 255, 255
    flattened = cv2.merge((r, g, b))

    return flattened


def perspective2(img):
    cropimg = img[130:240, :]

    src = np.array([
        [0, 0],
        [320, 0],
        [0, 110],
        [320, 110]
    ], dtype="float32")

    dst = np.array([
        [0, 0],
        [320, 0],
        [320 * 0.35, 110],
        [320 * 0.65, 110]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    wrapped = cv2.warpPerspective(cropimg, M, (320, 110))
    return wrapped

def sign_detect(self, img):

    showimg = img.copy()

    img = sign_flatten_rgb(img)  # 找到红色箭头标志，其余部分变白色

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 红色箭头灰度图
    if (DEBUG == True and DEBUG_SIGN_D == True):
        cv2.imshow("Gray", gray)

    dilation = preprocess(gray)
    if (DEBUG == True and DEBUG_SIGN_D == True):
        cv2.imshow("Preprocess", dilation)

    region, area = sign_find_region(dilation)
    if (DEBUG == True):
        self.show_ROI("Sign ROI", showimg, region)

    vision_history_len = len(self.vision_history_size)
    if (area == 0 and vision_history_len > 0):
        g0area.append(0)

    # during 10*100ms, car can not see anything, clean vision history
    if (vision_history_len > 0 and len(g0area) > 10):
        del self.vision_history_size[:]
        del self.vision_history[:]
        del g0area[:]
        if (DEBUG == True):
            print("clean vision history. %d" % len(g0area))
        return ""

    # Only condition is true, Do recognize.
    if (vision_history_len > 4 and len(g0area) > 3):
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
            print("max vision size = %d, vision history len = %d, See nothing = %d" % (
            self.vision_history_size[max_index], vision_history_len, len(g0area)))

        cropimg = self.vision_history[max_index]

        # remove border
        arr = Image.fromarray(cropimg)
        pilimg = arr.convert('L')
        invert_img = ImageOps.invert(pilimg)
        box = invert_img.getbbox()

        flagimg = cropimg[box[1]:box[3], box[0]:box[2]]

        del self.vision_history_size[:]
        del self.vision_history[:]
        del g0area[:]

        if (DEBUG == True and DEBUG_SIGN_D == True):
            cv2.imshow("Binary Image", flagimg)

        return self.recognize_sign(flagimg)

# how many tracks can we see in the camera
def how_many_tracks(img,track):
    which_track = track[8]

    func_mapping = {
        0: is_red,
        1: is_green,
        2: is_blue
    }

    if which_track == 4 or which_track == 3:
        return 6
    if which_track == 2:
        # in left blue, check right border if has track 4 color
        is_color = func_mapping[track[4]]
        for i in range(239, 70, -1):
            # means we can see track 4's color
            if is_color(img[i, 290]):
                return 6
        return 3
    if which_track == 5:
        # in right blue, check left border if has track 3 color
        is_color = func_mapping[track[3]]
        for i in range(239, 70, -1):
            if is_color(img[i, 30]):
                return 6
        return 3

    return 3


# color is not used in this version
def find_track(img, irow, color=2):
    left = -1
    right = 320

    track=self.track
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





def determine_middle(self, img, row_default=18, color=2):
    # ImageProcessor.show_image(img, "source")

    left, right = find_track(img, row_default, color=color)

    if right < 100:
        middle = 50
    elif left > 220:
        middle = 270
    else:
        middle = (right + left) / 2

    lost = False
    if right == 320 or left == -1:
        lost = True
    return middle, lost

if __name__ == '__main__':
    imgpath = './car2.jpg'
    img = Image.open('D:\My Project\IntelligentCar\TrendFormulaCar\s1.6\car3.jpg')
    oriimg = np.asarray(img)  # np_img.shape=(240,320,3)
    if (DEBUG == True and DEBUG_SIGN_D == True):
        cv2.imshow('ori img', oriimg)
        cv2.waitKey(0)

    flat_img = ImageProcessor.preprocess(oriimg)
    print(flat_img)
    if (DEBUG == True and DEBUG_SIGN_D == True):
        cv2.imshow('flat_img',flat_img)
        cv2.waitKey(0)

    track = ImageProcessor.which_color_and_track(flat_img) # (-1, 0, 2, 0, 1, 2, 1, -1, 3)
    track = list(track)
    print(track)


    filterimg = ImageProcessor.image_filter(flat_img)
    if (DEBUG == True and DEBUG_SIGN_D == True):
        cv2.imshow('filter image',filterimg)
        cv2.waitKey(0)

    track_count = how_many_tracks(filterimg, track)
    print(track_count)

    img = perspective2(filterimg)
    print("--------------")
    print(img)
    if (DEBUG == True and DEBUG_SIGN_D == True):
        # cv2.imshow('perspective image',img)
        # cv2.waitKey(0)
        green = (0, 255, 0)  # 4
        cv2.line(img, (110,20), (123, 5), green)  # 5
        cv2.imshow('perspective',img)
        cv2.waitKey()



    # imgupper = oriimg[0:90, 0:319].copy()
    # imgbottom = oriimg[110:200, 0:319].copy()
    #
    # # ImageProcessor.show_image(oriimg, "vision")
    # ImageProcessor.show_image(imgupper,"vision1")
    # # ImageProcessor.show_image(imgbottom, 'vision2')
    #
    # sign = Car.sign_detect(imgupper)
    # print("sign")
    # print(sign)
    #
    # # -1
    # showimg = imgupper.copy()


    # # 0
    # img = sign_flattern_rgb2(imgupper)
    # ImageProcessor.show_image(img, "after_sign_flattern_rgb")
    # # 1
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # if (DEBUG == True and DEBUG_SIGN_D == True):
    #     cv2.imshow('Gray',gray)
    #     cv2.waitKey(0)
    #
    #
    # # 2
    # dilation = preprocess(gray)
    # if (DEBUG == True and DEBUG_SIGN_D == True):
    #     cv2.imshow('dilation', dilation)
    #     cv2.waitKey(0)
    #
    # # 3
    # region, area = sign_find_region(dilation)
    # print("-------------region,area----------------")
    # print(region)
    # print(area)
    # if (DEBUG == True):
    #     Car.show_ROI("Sign ROI", showimg, region)
    #     cv2.waitKey(0)
    #
    #
