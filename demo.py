import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchcv.models.fpnssd.net import FPNSSD512
from torchcv.models.ssd import SSDBoxCoder
from torchcv.models.ssd.net import SSD512
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

torch.cuda.set_device(0)

def drawRectResults(img, res):
    img_cp = img.copy()
    results = res
    window_list = []
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3])//2
        h = int(results[i][4])//2
        cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,0,255),4)
        cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
        # cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        cv2.putText(img_cp,results[i][0],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        if results[i][0] == "car" or results[i][0] == "bus":
            window_list.append(((x-w,y-h),(x+w,y+h)))
    return img_cp


print('Loading model..')
net = FPNSSD512(num_classes=21)
net.load_state_dict(torch.load('./examples/ssd/fpnssd512_20_trained.pth'))
net.eval()
net.cuda()
print('start processing..')


#videocap = cv2.VideoCapture('/home/wynmew/workspace/VehicleDetection/examples/project_video.mp4')
videocap = cv2.VideoCapture('demo.mp4')
success, image = videocap.read()
img = Image.fromarray(image)
IW = img.width
IH = img.height
ow = oh = 512
img = img.resize((ow, oh))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
x = transform(img)
x = Variable(x, volatile=True)
x = x.cuda()
loc_preds, cls_preds = net(x.unsqueeze(0))
box_coder = SSDBoxCoder(net)
boxes, labels, scores = box_coder.decode(
    loc_preds.cpu().data.squeeze(), F.softmax(cls_preds.cpu().squeeze(), dim=1).data)
#print(labels, scores)

#image = Image.fromarray(np.roll(image,1,axis=-1))
image = Image.fromarray(image)
b, g, r = image.split()
image = Image.merge('RGB',(r,g,b))
draw = ImageDraw.Draw(image)
if (len(boxes) == 1):
    a = iter(list(range(boxes[0].shape[0])))
    for i in a:
        print('+' * 40)
        x1 = boxes[0][i][0] * (IW / ow)
        y1 = boxes[0][i][1] * (IH / oh)
        x2 = boxes[0][i][2] * (IW / ow)
        y2 = boxes[0][i][3] * (IH / oh)
        b = np.array([x1, y1, x2, y2])
        # print(b)
        draw.rectangle(list(b), outline='red')


fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(image)

videoname = 'PYVDOut.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter(videoname, fourcc, 20, (IW, IH))

def updatefig(*args):
    success, image = videocap.read()
    img = Image.fromarray(image)
    img = img.resize((ow, oh))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(img)
    x = Variable(x, volatile=True)
    x = x.cuda()
    loc_preds, cls_preds = net(x.unsqueeze(0))
    box_coder = SSDBoxCoder(net)
    boxes, labels, scores = box_coder.decode(
        loc_preds.cpu().data.squeeze(), F.softmax(cls_preds.cpu().squeeze(), dim=1).data)
    #print('*' * 60)
    #print(labels, scores)
    image = Image.fromarray(image)
    b, g, r = image.split()
    image = Image.merge('RGB', (r, g, b))
    draw = ImageDraw.Draw(image)
    #print(len(boxes))
    #print(boxes)
    if (len(boxes) == 1):
        a = iter(list(range(boxes[0].shape[0])))
        for i in a:
            #print('+' * 40)
            x1 = boxes[0][i][0] * (IW / ow)
            y1 = boxes[0][i][1] * (IH / oh)
            x2 = boxes[0][i][2] * (IW / ow)
            y2 = boxes[0][i][3] * (IH / oh)
            b = np.array([x1, y1, x2, y2])
            #print(b)
            draw.rectangle(list(b), outline='red')
    ax.clear()
    ax.imshow(image)
    oframe = np.array(image)
    oframe = oframe[:, :, ::-1].copy()
    outvideo.write(oframe)
    return ax


ani = animation.FuncAnimation(fig, updatefig, interval=1)
plt.show()

