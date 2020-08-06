import sys
work_dir = "/workspace/share/training/maskrcnn-benchmark/"
sys.path.append(work_dir)

import json
import urllib.request
from collections import OrderedDict
import numpy as np
import cv2
import os
from prepare_dataset.rotate import Rotate
from tqdm import tqdm 

l1 = ['scratch']

image_folder=os.path.join(work_dir,'damage/images/')
json_file=os.path.join(work_dir, 'damage/annotations/instances_train.json')

R=Rotate()
min_angle=-20
max_angle=27
step =7

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

with open(json_file) as f:
    data = json.load(f)

def draw(mask,color,image):
  points_x = mask[0][0::2]
  points_y = mask[0][1::2]
  for x,y in zip(points_x,points_y):
      cv2.circle(image, (int(x),int(y)), 2, color, -1)
  return image


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

data_all=data

m=1000000
i=1000000
print(len(data['images']))
for k in tqdm(range(len(data['images']))):
    info_image =data['images'][k]
    masks = [anno['segmentation'] for anno in data['annotations'] if (anno['image_id'] == info_image['id'])]
    url =info_image['scalabel_url']

    image = cv2.imread(image_folder+ info_image['file_name'])
    for angle in np.arange(min_angle,max_angle,step) :
        if(angle==0):
            continue
        dst = R.rotate_image(image, angle)
        new_mask, id_mask = R.rotate_masks(masks, angle)
        check=0
        for new_m in new_mask:
            annotations = OrderedDict()
            annotations['id'] = m
            m=m+1
            annotations['category_id'] = l1.index('scratch')
            annotations['iscrowd'] = 0
            annotations['segmentation'] = (np.array(new_m)).tolist()
            annotations['image_id'] = i
            annotations['area'] = PolyArea(np.array(new_m[0][0::2]), np.array(new_m[0][1::2]))
            bbox_x = min(new_m[0][0::2])
            bbox_y = min(new_m[0][1::2])
            bbox_w = max(new_m[0][0::2]) - min(new_m[0][0::2])
            bbox_h = max(new_m[0][1::2]) - min(new_m[0][1::2])
            if(bbox_w <6 or bbox_h <6) :
                continue
            annotations['bbox'] = (np.array([bbox_x, bbox_y, bbox_w, bbox_h])).tolist()

            data_all["annotations"].append(annotations)
            check = 1


        if check == 1:
            images = OrderedDict()
            images['id'] = i
            i=i+1
            images['license'] = 4
            images['coco_url'] = 'coco.org'
            images['flickr_url'] = 'flickr.org'
            images['scalabel_url'] = url.replace('/', '_').split(".jpg")[0] + "_"+str(angle) + ".jpg"
            h, w = dst.shape[:2]
            images['width'] = w
            images['height'] = h
            images['file_name'] = url.replace('/', '_').split(".jpg")[0] +"_" +str(angle) + ".jpg"
            images['date_captured'] = '2013-12-15 02:41:52'
            cv2.imwrite(image_folder +images['file_name'] ,dst)
            data_all['images'].append(images)
    

path_annt = os.path.join(workdir,"damage/annotations/instances_train.json")
with open(path_annt, 'w') as outfile:
    json.dump(data_all, outfile, indent=4, ensure_ascii=False)
