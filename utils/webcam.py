import cv2
import numpy as np
# cap = cv2.VideoCapture(0)  # 视频进行读取操作以及调用摄像头
# ret = cap.set(3, 640)
# ret = cap.set(4, 480)
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# while cap.isOpened():  # 判断视频读取或者摄像头调用是否成功，成功则返回true。
#     ret, frame = cap.read()
#     if ret is True:
#         cv2.imshow('frame', frame)
#
#     else:
#         break
#
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

class LoadWebcam:
    def __init__(self, img_size=[480,640], stride=32):
        self.img_size = img_size
        self.stride = stride
        self.cap = cv2.VideoCapture(0)  # video capture object

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        # img0 = cv2.flip(img0, 1)  # flip left-right

        if ret_val:
            cv2.imshow('frame', img0)
        img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        #
        return  img

    def __len__(self):
        return 0

dataset = LoadWebcam()
for img0 in dataset:
    print(img0.shape)