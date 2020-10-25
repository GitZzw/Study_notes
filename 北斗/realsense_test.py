import pyrealsense2 as rs


pipeline = rs.pipeline()
# 创建 config 对象：
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
while True: # 多取几幅图片，前几张不清晰
    # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
    frames = pipeline.wait_for_frames()
    print('wait for frames in the first loop')
    align_to_depth = rs.align(rs.stream.depth)
    align_to_depth.process(frames)
    get_depth_frame = frames.get_depth_frame()
    get_color_frame = frames.get_color_frame()

    if not get_color_frame and get_depth_frame: # 如果color和depth其中一个没有得到图像，就continue继续
        continue

    color_frame = np.asanyarray(get_color_frame.get_data())
    depth_frame = np.asanyarray(get_depth_frame.get_data())
    cv2.imshow("color", color_frame)
    cv2.imshow("depth", depth_frame)
