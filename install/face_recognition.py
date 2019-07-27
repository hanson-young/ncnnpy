import numpy as np
import ncnnpy
import cv2
import time

class FaceRecognition(object):
    def __init__(self,num_threads = 1, power_save = 0, threshold = 0.46):
        self._recog = ncnnpy.FaceRecognition()
        self._recog.init(num_threads, power_save)
        self.threshold = threshold

    def compare(self, feature1, feature2):
        num = np.dot(feature1, feature2)
        denom = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        cosine = num / denom
        # print(feature1)
        # print(feature2)
        print("compare score: %f (@ %f)"%(cosine,self.threshold))
        if cosine > self.threshold:
            return True
        else:
            return False
    def embedding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_T = img
        img_T = img.transpose((2, 0, 1)).flatten().reshape(3, 112, 112)
        img_T = img_T.astype(np.float32)
        img_T = (img_T - 127.5) / 128.0

        self._recog.inference(img_T)
        features = self._recog.get_features().copy()
        return features

    def example(self, face_path1, face_path2):
        face1 = cv2.imread(face_path1)
        face2 = cv2.imread(face_path2)
        start = time.time()
        f1 = self.embedding(face1)
        f2 = self.embedding(face2)
        print("Is same person --> ", self.compare(f1, f2))
        end = time.time()
        print("time:", end - start)


if __name__ == "__main__":
    face_recognition = FaceRecognition()
    print("##########################################")
    face_recognition.example("./id0_0.jpg","id0_1.jpg")
    print("##########################################")
    face_recognition.example("./id0_0.jpg","id1_0.jpg")
    print("##########################################")
    face_recognition.example("./id1_0.jpg","id1_1.jpg")
    print("##########################################")
    face_recognition.example("./id0_0.jpg","id1_1.jpg")
    print("##########################################")
