import json
import os.path as osp
import os
import sys
#from boxx import show
import cv2
import time
import numpy as np

def get_iou(a, b, epsilon=1e-5):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap
    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


ids = list()
counter = 0
label = []
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD512HaiRng20180813PlusBOX8000_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125ClassesTopRight_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_BOX8000_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_BOX3000_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125ClassesSel_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125ClassesSel_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_Aug_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_Aug2_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_Aug3_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_Aug4_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_Aug5_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCVVD', 'Results_evalFPNSSD5125Classes_Aug6_BOX_HaiRongTest')):
    ids.append((line.strip()))
'''
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCV', 'Results_evalFPNSSD5125Classes_Neg_HaiRongTest')):
    ids.append((line.strip()))
'''
for line in open(osp.join('/home/wynmew/workspace/PyTorchCV', 'Results_evalFPNSSD5125Classes_Neg_Aug_HaiRongTest')):
    ids.append((line.strip()))

for index in range(len(ids)):
    lineID = index
    if ids[lineID][0].isdigit():
        print(ids[lineID])
        if label != []:
            tmpid=-1
            maxscore=0
            for i in range(len(scores)):
                if scores[i] >maxscore:
                    maxscore = scores[i]
                    tmpid = i
            if tmpid != -1:
                if prdlabels[tmpid] == label:
                    iou = get_iou(pts,prdpts[tmpid])
                    print(iou)
                    counter +=1
        label=[]
        pts=[]
        prdpts = []
        prdlabels = []
        scores = []
    elif ids[lineID][0] == '+':
        continue
    elif ids[lineID][0] == 'G':
        #print(ids[lineID])
        tmp = ids[lineID].split(' ')
        label = int(tmp[5])
        pts = [float(tmp[1][1:-1]), float(tmp[2][:-1]), float(tmp[3][:-1]), float(tmp[4][:-1])]
    elif ids[lineID][0] == 'p':
        #print(ids[lineID])
        tmp = ids[lineID].split(' ')
        prdlabels.append(int(tmp[5]))
        prdpts.append([float(tmp[1][1:7]), float(tmp[2][:6]), float(tmp[3][:6]), float(tmp[4][:6])])
        scores.append(float(tmp[6][:6]))
    elif ids[lineID][0] == '-':
        #print('next')
        continue
    else:
        print('error1 at', lineID, " : ", ids[lineID])
        continue

if label != []:
    tmpid = -1
    maxscore = 0
    for i in range(len(scores)):
        if scores[i] > maxscore:
            maxscore = scores[i]
            tmpid = i
    if tmpid != -1:
        if prdlabels[tmpid] == label:
            iou = get_iou(pts, prdpts[tmpid])
            print(iou)
            counter += 1

print(counter)


