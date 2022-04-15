# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import time
from visualization import skeleton, keypoints
from loaddata import LoadWebcam, LoadImages
from utils import increment_path
from pathlib import Path

def run(weights="./models/movenet_singlepose_lightning_4", source=0, threshold=.35, vis_type='skeleton', exist_ok=False, save_res=False):
    video_writer = None
    if 'lightning' in weights:
        size = [192, 192]
    if 'thunder' in weights:
        size = [256, 256]
    # load model
    model = hub.load(weights)
    movenet = model.signatures['serving_default']

    save_dir = increment_path(Path('runs/detect') / 'demo')
    save_dir.mkdir(parents=True, exist_ok=exist_ok)  # make dir

    source = str(source)
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric()

    if webcam:
        dataset = LoadWebcam(size)
    else:
        dataset = LoadImages(source, img_size=size)

    for frame, frame0, path, vid_cap  in dataset:

        image = tf.cast(frame, dtype=tf.int32)
        # Run model inference.
        t1 = time.time()
        outputs = movenet(image)
        t2 = time.time()
        time_diff = t2 - t1

        # calculate fps
        fps = round(1 / time_diff)
        points = outputs['output_0']

        # calculate neck point only for visual
        left_should = points[0, 0, 5]
        right_should = points[0, 0, 6]
        neck = (left_should + right_should) / 2
        neck = neck[None, None, None, :]
        points = tf.concat([points, neck], axis=2)

        t3 = time.time()
        if vis_type == 'skeleton':
            drawed_img = skeleton(frame0, points, fps)
        else:
            drawed_img = keypoints(frame0, points, fps)
        t4 = time.time()
        post_process_time = t4 - t3
        print("draw image time: {} ms ".format(round(post_process_time,3)))
        if webcam:
            cv2.imshow('Movenet', drawed_img)

        if save_res:
            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)
            print(save_path)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, drawed_img)
            else:
                # 'video' or 'stream'
                # if isinstance(video_writer[0], cv2.VideoWriter):
                #     video_writer[0].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, drawed_img.shape[1], drawed_img.shape[0]
                    save_path += '.mp4'
                if not video_writer:
                    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            video_writer.write(drawed_img)

run(source='coolgirl.mp4', save_res=True)
# run(source='demo', save_res=True)



# ori_image = tf.io.read_file(image_path)
# ori_image = tf.image.decode_jpeg(ori_image)
# h, w, _ = ori_image.shape
# image = tf.expand_dims(ori_image, axis=0)
# input_image = tf.image.resize_with_pad(image, 192, 192)
# input_image = tf.cast(input_image, dtype=tf.int32)