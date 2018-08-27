import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchcv.transforms import resize
from torchcv.models.ssd import SSDBoxCoder
from torchcv.models.fpnssd import FPNSSD512
from PIL import Image
import numpy as np
import cv2

from concurrent import futures
import time
import grpc
import goods_pb2, goods_pb2_grpc

MyClassNum = 5
img_size = 512

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


def protobuf_decode(request_string):
    request = goods_pb2.GoodsReq()
    request = request_string
    samples = []
    for req_image in request.images:
        image = np.fromstring(req_image.data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        samples.append((image))
    return samples, request.request_id


class infer:
    def __init__(self, weightfile = 'ssdckpt_*.pth'):
        self.net = FPNSSD512(num_classes=MyClassNum + 1)
        self.net.load_state_dict(torch.load(weightfile)['net'])
        self.net.cuda()
        self.net.eval()
        self.box_coder = SSDBoxCoder(self.net)
        self.resp = []

    def PyInfer(self, encoded_sample, context):
        samples, request_ID = protobuf_decode(encoded_sample)
        net = self.net
        box_coder = self.box_coder
        response = goods_pb2.GoodsResp()
        response.request_id = request_ID
        for img, offset_low, offset_high, camera_pos in samples:
            imgPre = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(imgPre)
            img_pil_r = img_pil.resize((img_size, img_size))
            inputs = preprocess(img_pil_r)
            inputs.unsqueeze_(0)
            loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
            box_preds, label_preds, score_preds = box_coder.decode(
                loc_preds.cpu().data.squeeze(),
                F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                score_thresh=0.1)
            pre = response.goods_info.add()
            pre.camera_pos = camera_pos
            for idx in range(len(box_preds)):
                pts = box_preds[idx].tolist()[0]
                lbs = label_preds[idx].tolist()[0]
                scs = score_preds[idx].tolist()[0]
                # self.resp.append([camera_pos,idx,pts,lbs,scs])
                # print('p:', pts, lbs, scs)
                det = pre.detections.add()
                det.label = str(lbs)
                det.score = scs
                det.xmin = pts[0]
                det.ymin = pts[1]
                det.xmax = pts[2]
                det.ymax = pts[3]

        return response

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    goods_pb2_grpc.add_ImgInferServicer_to_server(infer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
