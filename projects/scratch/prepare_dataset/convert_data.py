import sys
import os
#work_dir = "/workspace/share/training/maskrcnn-benchmark/"
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

print(os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir)))


sys.path.append(work_dir)
#pass
import json
import cv2
import urllib.request
import numpy as np
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import argparse
import glob
import logging
import math
import random
from tqdm import tqdm
from detectron2.structures import BoxMode

l1 = ['scratch']

image_dataset_path = os.path.join(work_dir,"datasets/coco/images/")
coco_json_save_path = os.path.join(work_dir,"datasets/coco/annotations/")

dataturks_json_paths = glob.glob(os.path.join(work_dir,"datasets/dataturks/")+"*.json")
scalabel_json_paths = glob.glob(os.path.join(work_dir,"datasets/scalabel/")+"*.json")

split_path = os.path.join(work_dir,"datasets/")
#print(image_dataset_path,coco_json_save_path)
#print(len(dataturks_json_paths),len(scalabel_json_paths))
#print(os.path.join(work_dir,"dataset/dataturks/*.json"))

def get_data_from_dataturks_json_folder(json_paths):
    data = []
    for json_path in json_paths:
        with open(json_path) as f:
            for line in f:
                line_data = json.loads(line)
                data.append(line_data)
    return data

def get_data_from_scalabel_json_folder(json_paths):
    data = []
    for json_path in json_paths:
        with open(json_path) as f:
            d1 = json.load(f)
            data.extend(d1)
    return data

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def check_none_in_annotation(poly2d_data):
    for poly in poly2d_data:
        if (not any(poly)):
            return  True
    return False

def get_bbox_from_mask(points_x,points_y):
    bbox_x = min(points_x)
    bbox_y = min(points_y)
    bbox_w = max(points_x) - min(points_x)
    bbox_h = max(points_y) - min(points_y)
    return [bbox_x,bbox_y,bbox_w,bbox_h]
    #return [min(points_x),min(points_y),max(points_x),max(points_y)]

def reformat_list_point(points,w,h):
    points_new = []
    points_x = []
    points_y = []
    for k in points:
        x = min(k[0] * w, w - 1)
        y = min(k[1] * h, h - 1)

        points_new.extend([x, y])
        points_x.append(x)
        points_y.append(y)
    return points_new,points_x,points_y

def get_image_contribute(img_id,url,image_name,w,h):
    images = OrderedDict()
    images['id'] = img_id
    images['license'] = 4
    images['coco_url'] = 'coco.org'
    images['flickr_url'] = 'flickr.org'
    images['scalabel_url'] = url
    images['width'] = w
    images['height'] = h
    images['file_name'] = image_name
    images['date_captured'] = '2013-12-15 02:41:52'
    return images

def get_annatation_contribute(img_id,anot_id,points_x,points_y,points_new):
    annotations = OrderedDict()
    annotations['id'] = anot_id
    annotations['category_id'] = 0
    annotations['iscrowd'] = 0
    annotations['segmentation'] = [points_new]
    annotations['image_id'] = img_id
    annotations['area'] = PolyArea(np.array(points_x), np.array(points_y))
    annotations['bbox'] = get_bbox_from_mask(points_x, points_y)
    annotations['bbox_mode'] = BoxMode.XYWH_ABS
    return annotations

def convert_dataturks_to_coco(data,anotation_id,image_id):
    annot_list = []
    image_list = []
    print("converting dataturks ...")
    for i in tqdm(range(len(data))):  # range for the url in json file
        check = 0
        try:

            image_name = data[i]['content'].replace('/', '_')
            label_data = data[i]['annotation']

            if label_data is None:
                continue

            for j in range(0, len(label_data)):

                if label_data[j]['label'][0] not in l1:
                    continue
                poly2d_data = label_data[j]['points']
                Polys_is_None = check_none_in_annotation(poly2d_data)

                if (len(poly2d_data) < 3 or (Polys_is_None)):
                    continue

                points = poly2d_data
                anotation_id = anotation_id + 1
                w = data[i]['annotation'][0]["imageWidth"]
                h = data[i]['annotation'][0]["imageHeight"]
                points_new, points_x, points_y=reformat_list_point(points,w,h)
                annotations = get_annatation_contribute(image_id,anotation_id,points_x,points_y,points_new)

                annot_list.append(annotations)
                check = 1

            if check == 1:
                image=cv2.imread(image_dataset_path+image_name)
                #imges=image.shape[:2]
                (h, w) = image.shape[:2]

                images=get_image_contribute(image_id, data[i]['content'],image_name,w,h)
                image_list.append(images)
                image_id=image_id +1
        
        except  Exception as e:
            logging.exception(e)
            continue

    return annot_list,image_list,anotation_id,image_id

def convert_scalabel_to_coco(data,anotation_id,image_id):
    annot_list = []
    image_list = []
    print("convert scalabel ...")
    for i in tqdm(range(len(data))):  # range for the url in json file
        check = 0
        try:
            image_name = data[i]['name'].replace('/', '_')
            label_data = data[i]['labels']
            if label_data is None:
                continue

            for j in range(0, len(label_data)):
                if label_data[j]['category'] not in l1:
                    continue
                poly2d_data = label_data[j]['poly2d']

                if len(poly2d_data) < 1:
                    continue
                #anotation_id = anotation_id + 1
                for pn in range(len(poly2d_data)):
                    points = poly2d_data[pn]['vertices']
                    points_new = []
                    points_x = []
                    points_y = []
                    for k in points:
                        points_new.extend(k)
                        points_x.append(k[0])
                        points_y.append(k[1])
                    annotations = get_annatation_contribute(image_id,anotation_id,points_x,points_y,points_new)
                    anotation_id += 1
                    annot_list.append(annotations)

                check = 1
            if check == 1:
                img = cv2.imread(image_dataset_path + image_name)
                (h, w) = img.shape[:2]
                images = get_image_contribute(image_id,data[i]['name'],image_name,w,h)
                image_list.append(images)
                image_id = image_id +1
    
        except Exception as e:
            logging.exception(e)
            continue
    return annot_list,image_list,anotation_id,image_id


def write_to_coco_json(image_list,annot_list,mode):
    out = OrderedDict()

    info = OrderedDict()
    info['description'] = 'Car Parts Dataset'
    info['url'] = 'http://cocodataset.org'
    info['version'] = '1.0'
    info['year'] = 2017
    info['contributor'] = 'COCO Consortium'
    info['date_created'] = '2017/09/01'

    licenses = OrderedDict()
    licenses['url'] = 'http://creativecommon.org/licences/by/2.0/'
    licenses['id'] = 4
    licenses['name'] = 'Attribution License'

    out['info'] = info
    out['licences'] = [licenses]

    list_cat = []
    for i in range(len(l1)):
        categories = OrderedDict()
        categories['supercategory'] = 'Carparts'
        categories['id'] = i
        categories['name'] = l1[i]
        list_cat.append(categories)

    out['images'] = image_list
    out['annotations'] = annot_list
    out['categories'] = list_cat

    path_annt = os.path.join(coco_json_save_path,"instances_"+ mode +".json")
    with open(path_annt, 'w') as outfile:
        json.dump(out, outfile, indent=4, ensure_ascii=False)

def unique(l):
    uni = []
    for i in l:
        if i['id'] in uni:
            #print('yo')
            l.remove(i)
        else:
            #print('nothing')
            uni.append(i['id'])

    return l

def main():
    anotation_id = 0
    image_id =0

    dataturk_data=get_data_from_dataturks_json_folder(dataturks_json_paths)
    scalabel_data=get_data_from_scalabel_json_folder(scalabel_json_paths)
    a1,i1,anotation_id,image_id = convert_dataturks_to_coco(dataturk_data,anotation_id,image_id)
    #print(anotation_id,image_id)
    a2, i2 ,_,_= convert_scalabel_to_coco(scalabel_data,anotation_id+1,image_id +1)
    
    #print(i2[10])
    print('-'*20)
    print(a2[10])
    #pass
    a1.extend(a2)
    i1.extend(i2)
    image_list=i1
    annot_list=a1

    train_set = np.load(split_path+'train_a0_1474.npy')
    valid_set = np.load(split_path +'valid_a0_400.npy')
    test_set  = np.load(split_path+'test_a0_400.npy')

    train_set = list(filter( lambda i:i['file_name'] in train_set,image_list ))
    train_set =  unique(train_set)
    #print(len(train_set))
    valid_set = list(filter( lambda i:i['file_name'] in valid_set,image_list ))
    valid_set = unique(valid_set)

    test_set = list(filter( lambda i:i['file_name'] in test_set,image_list ))
    test_set = unique(test_set)
    
    #valid_set = [i if (i['file_name'] in valid_set) for i in image_list ]
    #test_set = [i if (i['file_name'] in test_set) for i in image_list]

    print(len(train_set),len(valid_set),len(test_set))


    #random.shuffle(image_list)
    #print(image_list[0])
    write_to_coco_json(train_set,annot_list,"train")
    write_to_coco_json(valid_set,annot_list,"val")
    write_to_coco_json(test_set, annot_list, "test")

if __name__ == '__main__':
    main()












