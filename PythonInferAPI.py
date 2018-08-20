import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchcv.transforms import resize
from torchcv.models.ssd import SSDBoxCoder
from torchcv.models.fpnssd import FPNSSD512
from PIL import Image

MyClassNum = 5
net = FPNSSD512(num_classes=MyClassNum + 1)
net.load_state_dict(torch.load('ssdckpt_5Classes_BOX.pth')['net'])
net.cuda()
net.eval()

img_size = 512

def transform(img):
    img = resize(img, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])(img)
    return img

Img = '...'

box_coder = SSDBoxCoder(net)

img_pil = Image.open(Img)
img_pil_r = img_pil.resize((img_size,img_size))

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.ToTensor(),
   normalize
])

inputs = preprocess(img_pil_r)
inputs.unsqueeze_(0)

loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
box_preds, label_preds, score_preds = box_coder.decode(
    loc_preds.cpu().data.squeeze(),
    F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
    score_thresh=0.01)

for idx in range(len(box_preds)):
    pts = box_preds[idx].tolist()[0]
    lbs = label_preds[idx].tolist()[0]
    scs = score_preds[idx].tolist()[0]
    print('p:',pts, lbs, scs)
    print('--')

