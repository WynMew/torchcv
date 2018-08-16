import json
import os.path as osp
import sys

ids = list()

My_CLASSES = (  # always index 0
    '2guw', 'fm2a', 'nb5e', 'tbhp','xz7m',
 #   'eoqx' # neg
    )

class_to_ind = dict(zip(My_CLASSES, range(len(My_CLASSES))))

'''
for line in open(osp.join('/home/wynmew/workspace/Data', 'HaiRong20180813PlusBOX8000ImgList')):
    ids.append(('/home/wynmew/workspace/Data', line.strip()))
'''
for line in open(osp.join('/home/wynmew/workspace/Data', 'testSet')):
    ids.append(('/home/wynmew/workspace/Data', line.strip()))

for index in range(len(ids)-1):
    img_id = ids[index]
    annofile = osp.join(img_id[0], img_id[1]).replace("images", "annotations").replace('.jpg', '.json')
    label = annofile.split("/")[7]
    label_idx = class_to_ind[label]
    with open(annofile) as datafile:
        AnnoData = json.load(datafile)
    pts = AnnoData["annotations"][0]["shape"]
    # print(pts)
    xmin = AnnoData['width']
    ymin = AnnoData['height']
    xmax = 0
    ymax = 0
    for ptsidx in range(len(pts)):
        if xmin > pts[ptsidx]['x']:
            xmin = pts[ptsidx]['x']
        if ymin > pts[ptsidx]['y']:
            ymin = pts[ptsidx]['y']
        if xmax < pts[ptsidx]['x']:
            xmax = pts[ptsidx]['x']
        if ymax < pts[ptsidx]['y']:
            ymax = pts[ptsidx]['y']

    print(img_id[1], xmin, ymin, xmax, ymax, label_idx)


'''
xmin = AnnoData['width']
ymin = AnnoData['height']
xmax = 0
ymax = 0
for ptsidx in range(len(pts)):
    if xmin > pts[ptsidx]['x']:
        xmin = pts[ptsidx]['x']
    if ymin > pts[ptsidx]['y']:
        ymin = pts[ptsidx]['y']
    if xmax < pts[ptsidx]['x']:
        xmax = pts[ptsidx]['x']
    if ymax < pts[ptsidx]['y']:
        ymax = pts[ptsidx]['y']
'''