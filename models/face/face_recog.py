import dlib, cv2
import numpy as np


class face_recog():
    def __init__(self):
        self.img_paths={}
        self.registered = []
        try:
            self.descs=np.load("./models/face/descs.npy", allow_pickle=True).tolist()
            for key in self.descs.keys():
                self.registered.append(key)
        except:
            self.descs = {}
         #얼굴 탐지 모델
        self.detector=dlib.get_frontal_face_detector()
        #얼굴 랜드마크 탐지 모델
        self.sp=dlib.shape_predictor('./models/face/shape_predictor_68_face_landmarks.dat')
        #얼굴 인식 모델
        self.facerec=dlib.face_recognition_model_v1('./models/face/dlib_face_recognition_resnet_model_v1.dat')
        
    def find_faces(self, image):
            dets=self.detector(image, 1)   #얼굴 찾은 결과물이 들어감

            #얼굴 하나도 못 찾았을 경우 빈 배열 반환
            if len(dets)==0:
                return np.empty(0), np.empty(0), np.empty(0)

            rects, shapes=[], []
            shapes_np=np.zeros((len(dets), 68, 2), dtype=np.int)
            #얼굴의 갯수만큼 루프를 돔
            for k, d in enumerate(dets):
                rect=((d.left(), d.top()), (d.right(), d.bottom()))
                rects.append(rect)

                #68개의 점이 나옴. 랜드마크 찾기
                shape=self.sp(image, d)  

                for i in range(0, 68):
                    shapes_np[k][i]=(shape.part(i).x, shape.part(i).y)  

                shapes.append(shape)  

            return rects, shapes, shapes_np

    def encode_faces(self, image, shapes):
            face_decriptors=[]  #결과값 저장 리스트

            #랜드마크들의 배열 집합 크기만큼 루프
            for shape in shapes:
                #얼굴을 인코딩. 이 함수에는 전체 이미지와 랜드마크가 들어감
                face_descriptor= self.facerec.compute_face_descriptor(image, shape) 
                face_decriptors.append(np.array(face_descriptor))
            return np.array(face_decriptors)
            
    def register(self, name, path):
        self.img_paths[name]=path

        for name, img_path in self.img_paths.items():
            img_bgr=cv2.imread(img_path)
            #컬러 체계를 바꿈
            img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            #인코딩하는 함수에 전체 이미지와 각 사람의 랜드마크를 넣어줌
            #인코딩된 결과를 각 사람의 이름에 맞게 저장
            _, img_shapes, _= self.find_faces(img_rgb)
            if (len(img_shapes) == 0):
                return -1
            self.descs[name] = self.encode_faces(img_rgb, img_shapes)[0]
        np.save('./models/descs.npy', self.descs)
        print("npy saved.")

        #print(descs)

    def recognize(self, find):
        #Compute Input
        #단체 사진에서 얼굴을 찾음
        img_bgr=cv2.imread(find)
        img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        rects, shapes, _ = self.find_faces(img_rgb)
        if (len(shapes) == 0):
                return -1
        descriptors = self.encode_faces(img_rgb, shapes)

        for i, desc in enumerate(descriptors):
            found=False
            for name, saved_desc in self.descs.items():
                #두 벡터 사이의 유클리드 거리를 구함
                dist=np.linalg.norm([desc]-saved_desc, axis=1)
                if dist<0.4:
                    return name
        return "등록되지 않은 사람입니다."


