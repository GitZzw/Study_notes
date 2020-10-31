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
import numpy as np
import pyrealsense2 as rs
import cv2
from openvino.inference_engine import IENetwork, IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

fx = 611.855712890625
fy = 611.8430786132812
cx = 317.46136474609375
cy = 247.88717651367188

plane_x = 0.0
plane_y = 0.0
plane_z = 0.0



def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
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
    return parser


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


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

def find_min_box(obj_list_,color_frame_,depth_frame_):
    global plane_x,plane_y,plane_z
    box_min = 100000000
    obj_min = obj_list_[0]
    #find min box
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
    #args = build_argparser().parse_args()

    #model_xml = args.model
    #model_bin = os.path.splitext(model_xml)[0] + ".bin"
    global fx, fy, cx, cy, plane_x, plane_y, plane_z
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.cpu_extension = False
    args.model = 'frozen_darknet_yolov3_model.xml'
    args.bin = 'frozen_darknet_yolov3_model.bin'
    args.device = 'MYRIAD'
    args.labels = 'frozen_darknet_yolov3_model.mapping'
    args.input = 'outdoorsd435.avi'
    args.prob_threshold = 0.6
    args.iou_threshold = 0.6
    args.raw_output_message = True
    args.no_show = False


    #TCP IP SETUP
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(('127.0.0.1',8000))
    server.listen(5)
    print("waiting msg ...")
    conn, clint_add = server.accept()

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------

    log.info("Loading network")
    net = IENetwork(model=args.model, weights=args.bin)


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

    is_async_mode = True
    number_input_frames = -1
    wait_key_code = 1

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    if number_input_frames != 1:
        frame = pipeline.wait_for_frames()

        # 对齐时间戳
        align_to_depth = rs.align(rs.stream.depth)
        align_to_depth.process(frame)
        #深度和rgb
        get_next_frame = frame.get_color_frame()
        get_depth_frame = frame.get_depth_frame()
        depth_frame = np.asanyarray(get_depth_frame.get_data())
        frame = np.asanyarray(get_next_frame.get_data())
    else:
        is_async_mode = False
        wait_key_code = 0

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)

    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")
    while True:
        # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
        # in the regular mode, we capture frame to the CURRENT infer request
        if is_async_mode:
            next_frame = pipeline.wait_for_frames()
            align_to_depth.process(next_frame)
            get_next_frame = next_frame.get_color_frame()
            get_next_depth_frame = next_frame.get_depth_frame()

            next_depth_frame = np.asanyarray(get_next_depth_frame.get_data())
            next_frame = np.asanyarray(get_next_frame.get_data())

        else:
            frame = pipeline.wait_for_frames()

            # 对齐时间戳
            align_to_depth = rs.align(rs.stream.depth)
            align_to_depth.process(frame)
            #深度和rgb
            get_next_frame = frame.get_color_frame()
            get_depth_frame = frame.get_depth_frame()
            depth_frame = np.asanyarray(get_depth_frame.get_data())
            frame = np.asanyarray(get_next_frame.get_data())

        if not get_next_frame:
            continue


        print("next_frame")
        if is_async_mode:
            request_id = next_request_id
            in_frame = cv2.resize(next_frame, (w, h))
        else:
            request_id = cur_request_id
            in_frame = cv2.resize(frame, (w, h))

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
                                             frame.shape[:-1], layer_params,
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

        origin_im_size = frame.shape[:-1]

        #存储识别到的obj
        obj_list = list()


        for obj in objects:
            # Validation bbox of detected object
            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                continue
            color = (255,255,0)
            det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                str(obj['class_id'])

            if args.raw_output_message:
                log.info(
                    "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                              obj['ymin'], obj['xmax'], obj['ymax'],
                                                                              color))

            #cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            # cv2.putText(frame,
            #             "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
            #             (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            # #添加obj
            obj_list.append(obj)


        if len(obj_list) > 0:
            objmin = find_min_box(obj_list,color_frame,depth_frame)
            cv2.putText(color_frame,
                    "#" + det_label + ' ' + str(round(objmin['confidence'] * 100, 1)) + ' %',
                    (objmin['xmin'], objmin['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        # TCP/IP
        #TCP/IP 1 receive
        print("receive msg ...")
        data = conn.recv(5).decode('utf-8')


        plane_x = 10
        plane_y = 0
        plane_z = 0
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
            conn.send(send_data_byte)



        # Draw performance stats over frame
        inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
            "Inference time: {:.3f} ms".format(det_time * 1e3)
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
        async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
            "Async mode is off. Processing request {}".format(cur_request_id)
        parsing_message = "YOLO parsing time is {:.3f}".format(parsing_time * 1e3)

        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        # cv2.putText(frame, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
        #             (10, 10, 200), 1)
        cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        start_time = time()
        cv2.imshow("DetectionResults", frame)
        render_time = time() - start_time

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        key = cv2.waitKey(wait_key_code)


        # ESC key
        if key == 27:
            break
        # Tab key
        if key == 9:
            exec_net.requests[cur_request_id].wait()
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
