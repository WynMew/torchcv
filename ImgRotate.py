import json
import os.path as osp
import sys
import os
import imutils
import cv2
from boxx import show
import numpy as np
import random

ids = list()

My_CLASSES = (  # always index 0
    '2guw', 'fm2a', 'nb5e', 'tbhp', 'xz7m',
    '6ate', '6uxf', 'h8u5',
    '6cw5', 'h1eu', 'kwy9'
 #   'eoqx' # neg
    )

class_to_ind = dict(zip(My_CLASSES, range(len(My_CLASSES))))

for line in open(osp.join('/home/wynmew/workspace/Data', 'Data20180928ImgAvailableList')):
    ids.append(('/home/wynmew/workspace/Data', line.strip()))

for index in range(len(ids)):
    if (index % 3 ==0):
        img_id = ids[index]
        imgfile = osp.join(img_id[0], img_id[1])
        annofile = osp.join(img_id[0], img_id[1]).replace("images", "annotations").replace('.jpg', '.json')
        pts = []
        with open(annofile) as datafile:
            AnnoData = json.load(datafile)
            pts = AnnoData["annotations"][0]["shape"]

        img =  cv2.imread(imgfile)
        angle = random.uniform(0, 350)
        #rotated = imutils.rotate_bound(img,angle)
        (h, w) = img.shape[:2]
        (cX, cY) = (w / 2, h / 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotated = cv2.warpAffine(img, M, (nW, nH))

        xmin = AnnoData['width']
        ymin = AnnoData['height']
        xmax = 0
        ymax = 0
        for ptsidx in range(len(pts)):
            amtx = np.asarray([(pts[ptsidx]['x'], pts[ptsidx]['y'])])
            rot=amtx.dot(M[:, :2].transpose()) + M[:, 2].transpose()
            if xmin > rot[0,0]:
                xmin = rot[0,0]
            if ymin > rot[0,1]:
                ymin = rot[0,1]
            if xmax < rot[0,0]:
                xmax = rot[0,0]
            if ymax < rot[0,1]:
                ymax = rot[0,0]

        label = annofile.split("/")[7]
        label_idx = class_to_ind[label]
        print(img_id[1], xmin, ymin, xmax, ymax, label_idx)
        tmp=img_id[1].split('/')
        file=osp.join(img_id[0],'DataRotate',''.join(tmp[1:]))
        cv2.imwrite(file,rotated)