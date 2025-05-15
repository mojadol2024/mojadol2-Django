from django.shortcuts import render
import whisper
import tempfile
import ffmpeg
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse

model = whisper.load_model("base")
"""
_tts_instance = None

def get_tts():
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
    return _tts_instance
"""
class STTView(APIView):
    def post(self, request):
        video_file = request.FILES.get("audio")
        if not video_file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                for chunk in video_file.chunks():
                    temp_video.write(chunk)
                temp_video_path = temp_video.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio_path = temp_audio.name

            # ffmpeg로 mp4 → wav 변환
            ffmpeg.input(temp_video_path).output(temp_audio_path, ac=1, ar=16000, format='wav').overwrite_output().run(quiet=True)
            # whisper로 stt
            
            result = model.transcribe(temp_audio_path, language="ko")

            return Response({"text": result["text"]}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# 자소서 받기 -> 질문 생성 -> TTA오디오 생성 -> 오디오 db에 저장
# 질문에 대한 답변받기 -> STT변환 -> 답변에 대한 평가모델 -> 평가결과 db에 저장 -> PDF로 출력

# 프론트 처리 어떤가요? response time 3분이내 질문 20개만 1시간이내 서버컴 용량 부족 문제
"""
class TTSView(APIView):
    def post(self, request):
        text = request.data.get("text")
        if not text:
            return Response({"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            tts = get_tts()
            wav, sample_rate = tts.tts(text, speaker=tts.speakers[0])  # numpy 배열로 음성 생성

            buf = io.BytesIO()
            sf.write(buf, wav, samplerate=sample_rate, format='WAV')
            buf.seek(0)

            return HttpResponse(buf.read(), content_type="audio/wav")

        except Exception as e:
            print("===== TTS 오류 발생 =====")
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
"""