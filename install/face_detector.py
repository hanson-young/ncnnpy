import numpy as np
import ncnnpy
import cv2
import time

class FaceDetector(object):
    def __init__(self,num_threads = 1, power_save = 0, minface = 80):
        self._detector = ncnnpy.FaceDetector()
        self._detector.init(num_threads, power_save, minface)

    def get_allfaces(self, img):
        """Implementation based on Retinaface"""
        start = time.time()
        img_T = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_T = img_T.transpose((2,0,1)).flatten().reshape(img_T.shape[2],img_T.shape[0],img_T.shape[1])
        img_T = img_T.astype(np.float32)
        start = time.time()
        self.face_info = self._detector.get_allface(img_T)
        end = time.time()
        print("time:",end - start)
        return self.face_info

    def get_maxface(self, img):
        """Implementation based on mtcnn"""

        start = time.time()
        img_T = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_T = img.transpose((2,0,1)).flatten().reshape(img_T.shape[2],img_T.shape[0],img_T.shape[1])
        img_T = img_T.astype(np.float32)
        self.face_info = self._detector.get_maxface(img_T)
        end = time.time()
        print("time:",end - start)
        return self.face_info

    def results_show(self, title, img):
        for i in range(self.face_info.shape[0]):
            cv2.rectangle(img, (int(self.face_info[i][1]), int(self.face_info[i][2])), (int(self.face_info[i][3]), int(self.face_info[i][4])), (255, 0, 0), 2)
            for j in range(5):
                cv2.circle(img, (int(self.face_info[i][2*j + 5]), int(self.face_info[i][2 * j + 6])), 2, (0, 255, 255), 2)
        cv2.imwrite(title + ".jpg", img)
        cv2.imshow(title, img)
        cv2.waitKey(0)

if __name__ == "__main__":

    img = cv2.imread(r"./test.jpg")
    face_detector = FaceDetector()
    print("############### get all faces #################")
    face_detector.get_allfaces(img)
    face_detector.results_show("all faces", img.copy())

    print("############### get max face #################")
    face_detector.get_maxface(img)
    face_detector.results_show("max face", img.copy())

