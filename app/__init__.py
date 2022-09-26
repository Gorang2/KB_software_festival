from datetime import datetime
from flask import request
from flask import Flask
from flask import jsonify
import sys, os

from app.handle_audio import handle_audio
from app.handle_img import handle_img

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.nlp.voice_recog import NLP
from models.expression.face_exp_recog import face_exp
from models.face.face_recog import face_recog


def create_app():
    app = Flask(__name__)
    # 자연어처리 객체
    nlp = NLP()
    # 얼굴인식 객체
    face_rec = face_recog()
    # 표정인식 객체
    face_exps = face_exp()
    print('\n\nServer Initialized.')

    @app.route('/', methods=['POST', 'GET'])
    def post():
            if (request.method == 'POST'):
                print("POST : {}".format(datetime.now()))
                
                print('recieved file : {}'.format(request.files))
                # 오디오 파일 처리
                if 'audio' in request.files:
                    res = handle_audio(request, nlp)
                if 'image' in request.files:
                    res = handle_img(request, face_exps, face_rec)
                return jsonify(res)
            else :
                return "Server Activated."
    
    
    return app
