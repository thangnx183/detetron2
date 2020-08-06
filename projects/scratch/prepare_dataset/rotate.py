import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

class Rotate(object):
    def __int__(self):
        self.M = None
        self.w = 0
        self.h = 0
        self.dw = 0
        self.dh = 0

    def rotate_image(self, img, angle):
        height, width = img.shape[:2]
        self.M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        dst = cv2.warpAffine(img, self.M, (width, height))
        self.w, self.h = self.rotatedRectWithMaxArea(width, height, angle)
        crop_img = dst[int(height / 2 - self.h / 2):int(height / 2 + self.h / 2),int(width / 2 - self.w / 2):int(width / 2 + self.w / 2)]
        self.dw = (width - self.w)
        self.dh = (height - self.h)
        #image = cv2.resize(crop_img, (width, height), interpolation=cv2.INTER_AREA)
        return crop_img

    def rotate_masks(self, masks,angle):
        new_mask = []
        id_mask = []

        for i, mask in enumerate(masks):
            points_list = self.mask_transforms(mask,angle)
            if (len(points_list[0]) < 6):
                continue
            id_mask.append(i)
            new_mask.append(points_list)
        return new_mask, id_mask

    def rotatedRectWithMaxArea(self, w, h, angle):
        angle = (math.pi) * angle / 180

        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h

        side_long, side_short = (w, h) if width_is_longer else (h, w)
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))

        if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
        return wr, hr

    def mask_transforms(self, points_list,angle):
        _points_list =[]
        points_x = points_list[0][0::2]
        points_y = points_list[0][1::2]

        for x,y in zip(points_x,points_y):
            t_point = [int(x),int(y)]
            _points_list.append(t_point)

        mask_img = self.create_mask([_points_list],angle)

        contours=self.find_contour(mask_img)
        new_points_list=[]
        for con in contours:
            point=[con[0][0],con[0][1]]
            new_points_list.extend(point)
        return [new_points_list]

    def create_mask(self,polys,angle):
        mask = np.zeros((int(self.dh+self.h), int(self.dw+self.w)), dtype=np.int8)
        cv2.fillPoly(mask,  np.array(polys), 255)
        mask_gray = cv2.normalize(src=mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8UC1)
        mask_gray=self.rotate_image(mask_gray,angle)
        return mask_gray

    def find_contour(self,imgray):
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contours)==0):
            return []
        return contours[0]

