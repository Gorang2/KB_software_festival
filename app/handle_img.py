from datetime import datetime
import os

def handle_img(request, face_exps, face_rec):
    print("Image detected.")
    try:
                img = request.files['image']
                # 이미지 저장
                curtime = str(datetime.now()).split(' ')[1].split('.')[0]
                img_path = './images/{}.jpeg'.format(curtime)
                img.save(img_path)
                print("img saved : {}".format(img_path))

                # 등록 세션이라면 ->
                if 'text' in request.files:
                    name = request.files['text'].filename.replace(".txt", "")
                    print(name)
                    if name in face_rec.registered:
                        print("Already Recognized.")
                        res = {'statusCode' : -2, 'msg' : "Face already recognized", 'result' : '아직은 데이터가 부족합니다.', 'name' : name }
                        return res
                    
                    check_regi = face_rec.register(name, img_path)
                    if (check_regi == -1):
                        print("Face not recognized in photo.")
                        res = {'statusCode' : -1, 'msg' : 'Face not recognized', 'result' : '아직은 데이터가 부족합니다.', 'name' : "사진에 얼굴이 없습니다."}
                        return res
                    else:
                        print("Face recognized")
                    res = {'statusCode' : 200, 'msg' : "Face recognized : {}".format(name)}
                    return res

                # 표정분석 결과
                output_exp = face_exps.predict(img_path)
                print("expression result : {}".format(output_exp))

                # 얼굴인식 결과
                output_face = face_rec.recognize(img_path)
                if (output_face == -1):
                    output_face = '얼굴인식에 실패했습니다.'
                print("output face : {}".format(output_face))
                color = 'FFFFFF'
                if (output_exp == 'happy'):
                    color = 'FFFF00'
                elif (output_exp == 'anger'):
                    color = 'FF0000'
                elif (output_exp == 'fear'):
                    color = '9370DB'
                elif (output_exp == 'disgust'):
                    color = '008000'
                elif (output_exp == 'neutral'):
                    color = 'FFFFFF'
                elif (output_exp == 'sad'):
                    color = '0000FF'
                elif (output_exp == 'surprise'):
                    color = 'FFA500'
                 # 결과 반환
                res = {'statusCode' : 200, 'msg' : 'Image upload completed : {}'.format(img.filename), 'result' : output_exp, 'name' : output_face, 'color' : color, 'time' : curtime}
                return res
            # 이미지 처리 에러
    except Exception as e:
                print("Error in handling image", e)
                res = {'statusCode' : -1, 'msg' : 'Error in image upload'}
                return res
            