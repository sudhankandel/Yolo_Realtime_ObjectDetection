import numpy as np
import os
import cv2 as cv
import argparse
from yolo_utils import infer_image

FLAGS=[]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weights',
        type=str,
        default='./yolo_file/yolov3-tiny.weights',
        help='Path to the file which contains the weights \
                for YOLOv3.')

    parser.add_argument('-cfg', '--config',
        type=str,
        default='./yolo_file/yolov3-tiny.cfg',
        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-v', '--video-path',
        type=str,
        help='The path to the video file')


    parser.add_argument('-vo', '--video-output-path',
        type=str,
        default='./output.mp4',
        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./labels/coco-labels',
        help='Path to the file having the \
                    labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    FLAGS, unparsed = parser.parse_known_args()

    labels = open(FLAGS.labels).read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    if FLAGS.video_path is None:
        print ('Path to video not provided')
    elif FLAGS.video_path:
        vid = cv.VideoCapture(str(FLAGS.video_path))
        height, width, writer= None, None, None
        while True:

            grabbed, frame = vid.read()

            if not grabbed:
                break

            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

            if writer is None:
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,(frame.shape[1], frame.shape[0]), True)


            writer.write(frame)


        print ("[INFO] Cleaning up...")
        writer.release()
        vid.release()


    else:
        print("[ERROR] Something's not right...")