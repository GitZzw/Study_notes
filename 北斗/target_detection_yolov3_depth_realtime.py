#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time

import cv2
from openvino.inference_engine import IECore
import numpy as np

import pyrealsense2 as rs
import socket
import struct

# wzy global variable start >>>
enable_pnp = False
objectPts = np.zeros([4,2],dtype=np.float)
imgPts = np.zeros([4,2],dtype=np.int)
cameraMatrix = np.zeros([3,3],dtype=np.float)
distCoeffs = np.zeros((5), dtype=np.float)
outputRvecRaw = np.zeros((3), dtype=float)
outputTvecRaw = np.zeros((3), dtype=float)

left_glo = np.array([0,0])
right_glo = np.array([0,0])
up_glo = np.array([0,0])
down_glo = np.array([0,0])

fx = 611.855712890625
fy = 611.8430786132812
cx = 317.46136474609375
cy = 247.88717651367188

# wzy global variable end <<<

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()
# -------------------------------- tcp configure ----------------------------- #
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('127.0.0.1',8000))
server.listen(5)
print("waiting msg ...")
conn, clint_add = server.accept()
plane_x = 0.0
plane_y = 0.0
plane_z = 0.0

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, default='frozen_darknet_yolov3_model.xml', type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, default='cam', type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="MYRIAD", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
    return parser


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 1 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def get_contours_center(img):
    contours, hierarchy= cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x_sum = 0
    y_sum = 0
    success = False
    if len(contours) == 1:
        data = np.squeeze(contours)
        print(data)
        for i in data:
            x,y = i
            # print('x: ', x,'   y: ,', y)
            x_sum = x_sum + x
            y_sum = y_sum + y
        x_center = int(x_sum*1.0/len(data))
        y_center = int(y_sum*1.0/len(data))
        print('x_center: ', x_center, 'y_center: ', y_center)
        # if len(contours) == 1:
        success = True
        return success,x_center,y_center

    else:
    	x_center = 0
    	y_center = 0
    	return success,x_center,y_center


def find_img_pts(src,x_corner,y_corner):
    global left_glo,  right_glo,  up_glo, down_glo, enable_pnp
    print('shape: ', src.shape)
    height,width,channel = src.shape
    center = np.array([int(height/2.0),int(width/2.0)]) # height y first ,width x second
    print(center)

    hori_left_up_corner = np.array([int(center[0]-height/5.0),0])
    verti_left_up_corner = np.array([0,int(center[1]-width/5.0)])
    hori_left_band = src[int(center[0]-height/5.0):int(center[0]+height/5.0), 0:int(width/2.0)]
    hori_right_band = src[int(center[0]-height/5.0):int(center[0]+height/5.0), int(width/2.0):width]
    verti_up_band = src[0:int(height/2.0) , int(center[1]-width/5.0):int(center[1]+width/5.0)]
    verti_down_band = src[int(height/2.0):height , int(center[1]-width/5.0):int(center[1]+width/5.0)]

    hori_left_hsv = cv2.cvtColor(hori_left_band,cv2.COLOR_BGR2HSV)
    hori_right_hsv = cv2.cvtColor(hori_right_band,cv2.COLOR_BGR2HSV)
    verti_up_hsv = cv2.cvtColor(verti_up_band,cv2.COLOR_BGR2HSV)
    verti_down_hsv = cv2.cvtColor(verti_down_band,cv2.COLOR_BGR2HSV)
    hori_lower_hsv = np.array([0,50,50])
    hori_upper_hsv = np.array([10,150,150])
    verti_lower_hsv = np.array([0,50,50])
    verti_upper_hsv = np.array([10,150,150])

    hori_left_mask = cv2.inRange(hori_left_hsv, hori_lower_hsv, hori_upper_hsv)
    hori_right_mask = cv2.inRange(hori_right_hsv, hori_lower_hsv, hori_upper_hsv)
    verti_up_mask = cv2.inRange(verti_up_hsv, verti_lower_hsv, verti_upper_hsv)
    verti_down_mask = cv2.inRange(verti_down_hsv, verti_lower_hsv, verti_upper_hsv)

    # erode and dilate
    erode_kernel = np.ones((2, 2), np.uint8)
    hori_left_erosion = cv2.erode(hori_left_mask, erode_kernel, iterations=1)
    hori_right_erosion = cv2.erode(hori_right_mask, erode_kernel, iterations=1)
    verti_up_erosion = cv2.erode(verti_up_mask, erode_kernel, iterations=1)
    verti_down_erosion = cv2.erode(verti_down_mask, erode_kernel, iterations=1)
    #cv2.imshow('hori_erosion', hori_erosion)
    #cv2.imshow('verti_erosion', verti_erosion)

    dilate_kernel = np.ones((5, 5), np.uint8)
    hori_left_dilate = cv2.dilate(hori_left_erosion, dilate_kernel, iterations=1)
    hori_right_dilate = cv2.dilate(hori_right_erosion, dilate_kernel, iterations=1)
    verti_up_dilate = cv2.dilate(verti_up_erosion, dilate_kernel, iterations=1)
    verti_down_dilate = cv2.dilate(verti_down_erosion, dilate_kernel, iterations=1)
    cv2.imshow('hori_left_dilate', hori_left_dilate)
    cv2.imshow('hori_right_dilate', hori_right_dilate)
    cv2.imshow('verti_up_dilate', verti_up_dilate)
    cv2.imshow('verti_down_dilate', verti_down_dilate)



    left_state,left_x,left_y = get_contours_center(hori_left_dilate)
    right_state,right_x,right_y = get_contours_center(hori_right_dilate)
    up_state,up_x,up_y = get_contours_center(verti_up_dilate)
    down_state,down_x,down_y = get_contours_center(verti_down_dilate)
#    left_state,left_x,left_y = get_contours_center(hori_left_mask)
#    right_state,right_x,right_y = get_contours_center(hori_right_mask)
#    up_state,up_x,up_y = get_contours_center(verti_up_mask)
#    down_state,down_x,down_y = get_contours_center(verti_down_mask)

#    cv2.imshow('hori_left_mask', hori_left_mask)
#    cv2.imshow('hori_right_mask', hori_right_mask)
#    cv2.imshow('verti_up_mask', verti_up_mask)
#    cv2.imshow('verti_down_mask', verti_down_mask)
    cv2.waitKey(1)

    print(left_state,'y: ',left_y,' x: ',left_x)
    print(right_state,'y: ',right_y,' x: ',right_x)
    print(up_state,'y: ',up_y ,' x: ',up_x )
    print(down_state,'y: ',down_y ,' x: ',down_x)
    # calculate pixels of the origin image
    left_glo = np.array([x_corner+left_x, y_corner+int(center[0]-height/5.0)+left_y])
    right_glo = np.array([x_corner+int(width/2.0)+right_x, y_corner+int(center[0]-height/5.0)+right_y])
    up_glo = np.array([x_corner+int(center[1]-width/5.0)+up_x, y_corner+right_y])
    down_glo = np.array([x_corner+int(center[1]-width/5.0)+down_x, y_corner+int(height/2.0)+down_y])

    if left_state or right_state or up_state or down_state:
        print('detect target , number of marker is wrong!!!')
        return True
    if left_state and right_state and up_state and down_state:
        print('!!! detect target  !!!  correct number of marker !!!')
        enable_pnp = True
        return True
    else:
        print('not detect target marker ......')
        return False


def track_apply(src,x_corner,y_corner): # x_corner,y_corner for the pixel of left_up corner of cropped img
    global left_glo, right_glo, up_glo, down_glo, enable_pnp, objectPts, imgPts, cameraMatrix, distCoeffs, outputRvecRaw, outputTvecRaw
    enable_pnp = False
    img_pts_success = find_img_pts(src,x_corner,y_corner)

    if img_pts_success:
        print('detect success')
    else:
        print('detect Pts fail')
    # if enable_pnp:
        #cv2.solvePnP(objectPts, imgPts, cameraMatrix, distCoeffs, outputRvecRaw, outputTvecRaw)
    # TO DO

def find_min_box(obj_list_,color_frame_,depth_frame_):
    global plane_x,plane_y,plane_z
    box_min = 100000000
    obj_min = obj_list_[0]
    for obj_ in obj_list_:
        box_area = abs((obj_['xmax'] - obj_['xmin'])*(obj_['ymax'] - obj_['ymin']))
        if box_area < box_min:
            box_min = box_area
            obj_min = obj_

    # get pose
    # wzy crop img to target size
            # crop_img = np.zeros((color_frame.shape[0],color_frame.shape[1]), dtype=np.uint8)
            # wzy for my transpose image
    crop_img = color_frame_.copy()

    cv2.rectangle(color_frame_, (obj_min['xmin'], obj_min['ymin']), (obj_min['xmax'], obj_min['ymax']), color=(100,150,200), thickness=2)

    w_min = obj_min['xmin']  if obj_min['xmin'] > 0 else 0
    w_max = obj_min['xmax'] if obj_min['xmax'] < 640 else 640
    h_min = obj_min['ymin']  if obj_min['ymin']  > 0 else 0
    h_max = obj_min['ymax']  if obj_min['ymax']  < 480 else 480
    #crop_img = crop_img[obj['xmin']:obj['xmax']+100, obj['ymin']:obj['ymax']+100]

    crop_img = crop_img[h_min:h_max, w_min:w_max] # first height, second width

    depth_box_width=obj_min['xmax']-obj_min['xmin']
    depth_box_height=obj_min['ymax']-obj_min['ymin']
    delta_rate=0.3
    x_box_min=int(obj_min['xmin']+depth_box_width*delta_rate)
    y_box_min=int(obj_min['ymin']+depth_box_height*delta_rate)
    x_box_max=int(obj_min['xmax']-depth_box_width*delta_rate)
    y_box_max=int(obj_min['ymax']-depth_box_height*delta_rate)
    #print(x_box_min)
    after_width=(depth_box_width*(1-2*delta_rate))
    after_height=(depth_box_height*(1-2*delta_rate))
    depth_crop_img_=depth_frame_[y_box_min:y_box_max,x_box_min:x_box_max]
    z_pos=depth_crop_img_.sum()/(after_width*after_height)*0.001
    #z_pos=sum(map(sum,bb))/(after_width*after_height)*0.001
    x_pos = (0.5 * (x_box_min + x_box_max) - cx) * z_pos / fx
    y_pos = (0.5 * (y_box_min + y_box_max) - cy) * z_pos / fy
    print('0.5 * (y_box_min + y_box_max):',0.5 * (y_box_min + y_box_max))
    print('(0.5 * (y_box_min + y_box_max) - cy):',(0.5 * (y_box_min + y_box_max) - cy))
    print('z_pos:',z_pos)
    print('x_pos:',x_pos)
    print('y_pos:',y_pos)
    # convert coordinate from camera to plane
    plane_x = -z_pos
    plane_y = -x_pos
    plane_z = y_pos

    cv2.resizeWindow("crop_image", 600, 600)
    cv2.imshow("crop_image", crop_img)

    return obj_min

def main():
    global fx, fy, cx, cy, plane_x, plane_y, plane_z
    # args = build_argparser().parse_args() # wzy change
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.cpu_extension = False
    args.model = 'frozen_darknet_yolov3_model.xml'
    args.bin = 'frozen_darknet_yolov3_model.bin'
    args.device = 'HDDL'
    args.labels = 'frozen_darknet_yolov3_model.mapping'
    args.input = 'outdoorsd435.avi'
    args.prob_threshold = 0.8
    args.iou_threshold = 0.6
    args.raw_output_message = True
    args.no_show = False

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network")
    # wzy change
    #net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")
    # net = ie.read_network('frozen_darknet_yolov3_model.xml', "frozen_darknet_yolov3_model.bin")
    net = ie.read_network(args.model,args.bin)

    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------

    log.info("Preparing inputs")
    input_blob = next(iter(net.inputs))

    #  Defaulf batch_size is 1
    net.batch_size = 1

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    # with open('frozen_darknet_yolov3_model.mapping', 'r') as f:
    #     labels_map = [x.strip() for x in f]

    # input_stream = 0 if args.input == "cam" else args.input
    #
    # is_async_mode = True
    # cap = cv2.VideoCapture(input_stream)
    #
    # number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames
    #
    # wait_key_code = 1
    #
    # # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
    # if number_input_frames != 1:
    #     ret, frame = cap.read()
    # else:
    #     is_async_mode = False
    #     wait_key_code = 0

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)

    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0

    # ----------------------------------------------5.1 realsense input video stream -----------------------------------
    # Configure depth and color streams
    pipeline = rs.pipeline()
    # 创建 config 对象：
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.infrared,1,640,480,rs.format.y8,30)
    # config.enable_stream(rs.stream.infrared,2,640,480,rs.format.y8,30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    get_img_flag = False
    count = 0
    while not (get_img_flag and count > 10): # 多取几幅图片，前几张不清晰
        # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
        frames = pipeline.wait_for_frames()
        print('wait for frames in the first loop')
        align_to_depth = rs.align(rs.stream.depth)
        align_to_depth.process(frames)
        get_depth_frame = frames.get_depth_frame()
        get_color_frame = frames.get_color_frame()
        # get_ir_frame_left = frames.get_infrared_frame(1)
        # get_ir_frame_right = frames.get_infrared_frame(2)

        if not get_color_frame and get_depth_frame: # 如果color和depth其中一个没有得到图像，就continue继续
            continue

        # color_profile = get_color_frame.get_profile()
        # cvsprofile = rs.video_stream_profile(color_profile)
        # color_intrin = cvsprofile.get_intrinsics()
        # color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]
        # print(color_intrin_part)
        color_frame = np.asanyarray(get_color_frame.get_data())
        depth_frame = np.asanyarray(get_depth_frame.get_data())
        get_img_flag = True # 跳出循环
        count += 1
    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")
    is_async_mode = True
    try:
        while True:
            # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
            # in the regular mode, we capture frame to the CURRENT infer request

            # >>>>>>>>>>>  calculate time sum >>>>>>>>>>>>>>>#
            cpu_start = time()
            print('--------------------------new loop---------------------------------')
            # if is_async_mode:
            #     ret, next_frame = cap.read()
            # else:
            #     ret, frame = cap.read()
            #
            # if not ret:
            #     break
            # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
            next_frames = pipeline.wait_for_frames()

            align_to_depth.process(next_frames)
            get_next_depth_frame = next_frames.get_depth_frame()
            get_next_color_frame = next_frames.get_color_frame()
            next_color_frame = np.asanyarray(get_next_color_frame.get_data())
            next_depth_frame = np.asanyarray(get_next_depth_frame.get_data())
            # cv2.imshow("color", next_color_frame)
            # cv2.imshow("depth", next_depth_frame)


            if is_async_mode:
                request_id = next_request_id
                in_frame = cv2.resize(color_frame, (w, h))
            else:
                request_id = cur_request_id
                in_frame = cv2.resize(color_frame, (w, h))

            # resize input_frame to network size
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))

            # Start inference
            start_time = time()
            exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})
            det_time = time() - start_time

            # Collecting object detection results
            objects = list()
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                output = exec_net.requests[cur_request_id].outputs
                start_time = time()
                for layer_name, out_blob in output.items():
                    out_blob = out_blob.reshape(net.layers[net.layers[layer_name].parents[0]].shape)
                    layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
                    log.info("Layer {} parameters: ".format(layer_name))
                    layer_params.log_params()
                    objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                                 next_color_frame.shape[:-1], layer_params,
                                                 args.prob_threshold)
                parsing_time = time() - start_time

            # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
            objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
            for i in range(len(objects)):
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
                        objects[j]['confidence'] = 0

            # Drawing objects with respect to the --prob_threshold CLI parameter
            objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]

            if len(objects) and args.raw_output_message:
                log.info("\nDetected boxes for batch {}:".format(1))
                log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

            origin_im_size = color_frame.shape[:-1]

            count = 1

            # TCP/IP part1
            print(clint_add)
            print("receive msg ...")
            data = conn.recv(5).decode('utf-8')
            print(len(data))
            plane_x = 10
            plane_y = 0
            plane_z = 0
            obj_list = list()

            for obj in objects:
                print('for obj count:')
                print(count)
                count = count + 1
                # Validation bbox of detected object
                if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                    continue
                color = (int(min(obj['class_id'] * 12.5, 255)),
                         min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])

                if args.raw_output_message:
                    log.info(
                        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                                  obj['ymin'], obj['xmax'], obj['ymax'],
                                                                                  color))
                obj_list.append(obj)



            if len(obj_list) > 0:
                objmin = find_min_box(obj_list,color_frame,depth_frame)
                cv2.putText(color_frame,
                        "#" + det_label + ' ' + str(round(objmin['confidence'] * 100, 1)) + ' %',
                        (objmin['xmin'], objmin['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            # TCP/IP part2
            if len(data)==5:
                print("now send msg ......")
                send_data = [int(plane_x*1000), int(plane_y*1000), int(plane_z*1000)]
                print(send_data)
                send_data_byte = bytes(0)
                for i in range(len(send_data)):
                    print(send_data[i])
                    senddata1000 = str(send_data[i])+','
                    print(senddata1000.encode())
                    send_data_byte += senddata1000.encode()

                print(send_data_byte)
                conn.send(send_data_byte)

            # Draw performance stats over frame
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1e3)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)
            parsing_message = "YOLO parsing time is {:.3f} ms".format(parsing_time * 1e3)

            cv2.putText(color_frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(color_frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(color_frame, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)
            cv2.putText(color_frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

            start_time = time()
            if not args.no_show:
                cv2.imshow("DetectionResults", color_frame)

            render_time = time() - start_time

            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                color_frame = next_color_frame
                depth_frame = next_depth_frame

            if not args.no_show:
                #key = cv2.waitKey(wait_key_code)
                key = cv2.waitKey(1)

                # ESC key
                if key == 27:
                    break
                # Tab key
                if key == 9:
                    exec_net.requests[cur_request_id].wait()
                    is_async_mode = not is_async_mode
                    log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

            cpu_end = time()
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>cpu time :  ', cpu_end-cpu_start)
    finally:
        pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
