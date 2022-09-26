# -*- coding: utf-8 -*-
"""Conference.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16KJPYk2cyM4wuTVHmUigooNwd3seMdeh
"""
MAX_SEQUENCE_LENGTH = 30
# Commented out IPython magic to ensure Python compatibility.

import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from googletrans import Translator
from transformers import Wav2Vec2ForCTC
from keras.models import load_model

class NLP():
    def __init__(self): 
        self.model_test = load_model('./models/nlp/best_weights.h5')
        #self.classes = ["neutral", "happy", "sad", "love","anger"]
        self.classes = ["neutral", "happy", "sad","anger", "hate"]
        self.transcription = ''
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.tokenizer = ElectraTokenizer.from_pretrained("bhadresh-savani/electra-base-emotion")
        self.model_predict = ElectraForSequenceClassification.from_pretrained("bhadresh-savani/electra-base-emotion")

    def audio_to_text(self, filepath):
        import speech_recognition as sr

        try:
            print("getting sr audiofile")
            korean_audio = sr.AudioFile(filepath)
            r = sr.Recognizer()
            print("Recognized audio")

            with korean_audio as source:
                audio = r.record(source)

            res = r.recognize_google(audio_data=audio, language='ko-KR')
            #res = r.recognize_google(audio_data=audio, language='en-US')
            print('voice converted to text : ', res)
            self.transcription = res
        except Exception as e:
            print(e)

    def predict(self):
        translator = Translator()
        res = translator.translate(self.transcription, src='ko', dest='en').text # 한영 변환을 새로운 칼럼 'eng'에 담습니다
        print(res)                                               # 기존 데이터 프레임에 한영 변환된 데이터프레임이 append 되어 리턴
        inputs = self.tokenizer(res, return_tensors="pt")

        with torch.no_grad():
            logits = self.model_predict(**inputs).logits

        predicted_class_id = logits.argmax().item()
        voice_output = self.model_predict.config.id2label[predicted_class_id]
        print('voice : ', voice_output)
        return voice_output