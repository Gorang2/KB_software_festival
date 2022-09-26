from datetime import datetime
from IPython.display import Audio
from pydub import AudioSegment

def handle_audio(request, nlp):
    print("audio file detected")
    curtime = str(datetime.now()).split(' ')[1].split('.')[0]
    try:
        audio = request.files['audio']
        file_name = 'audios/{}.m4a'.format(curtime)
        audio.save(file_name)
        Audio(file_name)

        m4a_file = file_name
        wav_filename = 'audios/{}.wav'.format(curtime)

        track = AudioSegment.from_file(m4a_file,  format= 'm4a')
        file_handle = track.export(wav_filename, format='wav')
        print("audio file saved : {}.".format(curtime))

        # 오디오 -> 텍스트 변환
        nlp.audio_to_text(wav_filename)
        # 변환한 텍스트로 감정분석
        prob = nlp.predict()
        print("voice result : {}".format(prob))
        # 결과 반환
        res = {'statusCode' : 200, 'msg' : 'audio upload completed.', 'result' : prob, 'time' : curtime}
        return res
            
            # 오디오에서 에러 날 시
    except Exception as e:
        print("Error in saving audio", e)
        res = {'statusCode' : -1, 'msg' : 'Error in audio upload.', 'time' : 'null'}
        return res