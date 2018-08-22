from PIL import Image
import numpy as np
import cv2

import grpc
import goods_pb2, goods_pb2_grpc

def protobuf_encode(samples):
    request = goods_pb2.GoodsReq()
    request.request_id = 'default'
    for data, offset_low, offset_high, camera_pos in samples:
        image = request.images.add()
        ### 传进来的是numpy格式的图像, 这里编码成png格式
        ### 这里采用png格式是为了后面的验证用的, 实际上编码格式是jpg
        image.data = (cv2.imencode('.png', data)[1]).tostring()
        image.offset_low = offset_low
        image.offset_high = offset_high
        image.camera_pos = camera_pos
    #return request.SerializeToString()
    return request

def protobuf_decode(request_string):
    request = goods_pb2.GoodsReq()
    request.ParseFromString(request_string)
    samples = []
    for req_image in request.images:
        ### protobuf中保存的是jpg格式的图像, 这里解码成numpy数组
        image = np.fromstring(req_image.data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        offset_low = req_image.offset_low
        offset_high = req_image.offset_high
        camera_pos = req_image.camera_pos
        samples.append((image, offset_low, offset_high, camera_pos))
    return samples, request.request_id

def run():
    samples_in = []

    Img = '/home/wynmew/workspace/Data/Hairong/images/2guw/bottom_left/goods_2guw_pos_bottom_left_time_1532596559863.jpg'
    imgCV2 = cv2.imread(Img)
    offset_low = 10
    offset_high = 100
    camera_pos = goods_pb2.LEFT_BOTTOM
    samples_in.append((imgCV2, offset_low, offset_high, camera_pos))

    Img = '/home/wynmew/workspace/Data/Hairong/images/xz7m/top_right/goods_xz7m_pos_top_right_time_1532596120092.jpg'
    imgCV2 = cv2.imread(Img)
    offset_low = 10
    offset_high = 100
    camera_pos = goods_pb2.RIGHT_TOP
    samples_in.append((imgCV2, offset_low, offset_high, camera_pos))

    ### 构建protobuf结构并序列化
    encoded = protobuf_encode(samples_in)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = goods_pb2_grpc.ImgInferStub(channel)
        response = stub.PyInfer(encoded)
    print(response)

if __name__ == '__main__':
    run()
