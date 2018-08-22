from PIL import Image
import numpy as np
import cv2

import grpc
import goods_pb2, goods_pb2_grpc

def protobuf_encode(samples):
    request = goods_pb2.GoodsReq()
    request.request_id = 'default'
    for data:
        image = request.images.add()
        image.data = (cv2.imencode('.png', data)[1]).tostring()
    return request

def run():
    samples_in = []

    Img = 'a.jpg'
    imgCV2 = cv2.imread(Img)
    samples_in.append((imgCV2))

    Img = 'b.jpg'
    imgCV2 = cv2.imread(Img)
    samples_in.append((imgCV2))

    ### 构建protobuf结构并序列化
    encoded = protobuf_encode(samples_in)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = goods_pb2_grpc.ImgInferStub(channel)
        response = stub.PyInfer(encoded)
    print(response)

if __name__ == '__main__':
    run()
