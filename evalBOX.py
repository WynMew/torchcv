import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchcv.transforms import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.models.ssd import SSD300, SSDBoxCoder
from torchcv.models.fpnssd import FPNSSD512

from PIL import Image

print('Loading model..')
MyClassNum = 5
# net = SSD300(num_classes=MyClassNum+1)
net = FPNSSD512(num_classes=MyClassNum + 1)
net.load_state_dict(torch.load('ssdckpt.pth')['net'])
net.cuda()
net.eval()

print('Preparing dataset..')
img_size = 512


def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    return img, boxes, labels


dataset = ListDataset(root='/home/wynmew/workspace/Data', \
                      list_file='8000listval',
                      transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
box_coder = SSDBoxCoder(net)

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

def eval(net, dataset):
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
        #gt_boxes.append(box_targets.squeeze(0))
        #gt_labels.append(label_targets.squeeze(0))
        loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.01)
        #pred_boxes.append(box_preds)
        #pred_labels.append(label_preds)
        #pred_scores.append(score_preds)
        tmp=[]
        tmp.append([box_targets.squeeze(0).tolist()[0],label_targets.squeeze(0).tolist()[0], 1])
        print('++++')
        print(box_targets.squeeze(0).tolist()[0], label_targets.squeeze(0).tolist()[0])
        for idx in range(len(box_preds)):
            pts = box_preds[idx].tolist()[0]
            lbs = label_preds[idx].tolist()[0]
            scs = score_preds[idx].tolist()[0]
            tmp.append([pts, lbs, scs])
            print(pts, lbs, scs)
            print('--')



eval(net, dataset)

'''
for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
    print('%d/%d' % (i, len(dataloader)))
    gt_boxes.append(box_targets.squeeze(0))
    gt_labels.append(label_targets.squeeze(0))
    loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
    box_preds, label_preds, score_preds = box_coder.decode(
        loc_preds.cpu().data.squeeze(),
        F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
        score_thresh=0.01)
    pred_boxes.append(box_preds)
    pred_labels.append(label_preds)
    pred_scores.append(score_preds)
    break

print(voc_eval(
    pred_boxes, pred_labels, pred_scores,
    gt_boxes, gt_labels, gt_difficults=None,
    iou_thresh=0.5, use_07_metric=True))
'''