from django.shortcuts import render
import whisper
import tempfile
import ffmpeg
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
import os
from moviepy import *
import librosa

model = whisper.load_model("base")
# 영상에서 오디오 추출
def extract_audio_from_video(video_path, audio_path="temp.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

# 오디오에서 텍스트 추출 (음성 인식)
def transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# WPM 계산
def calculate_wpm(text, audio_path):
    word_count = len(text.strip().split())
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    wpm = word_count / duration * 60
    return word_count, duration, wpm

# 속도 분류 및 피드백
def classify_speed(wpm):
    if wpm < 90:
        return "느림", "조금 더 빠르게 말해보세요. 당신을 응원합니다!"
    elif wpm <= 135:
        return "보통", "현재 속도가 적절합니다. 잘 하고 있어요!"
    else:
        return "빠름", "조금 더 천천히 말하면 전달력이 좋아집니다. 천천히 가도 됩니다!"

# 전체 분석 함수
def analyze_speaking_speed(video_path):
    audio_path = extract_audio_from_video(video_path)
    text = transcribe(audio_path)
    word_count, duration, wpm = calculate_wpm(text, audio_path)
    speed_label, feedback = classify_speed(wpm)

    os.remove(audio_path) 

    return {
        "text": text,
        "word_count": word_count,
        "duration_sec": round(duration, 2),
        "wpm": round(wpm, 2),
        "speed_label": speed_label,
        "feedback": feedback
    }
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
            result = analyze_speaking_speed(video_file)
            
            response_data = {}

            for key, value in result.items():
                response_data[key] = value

            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# 자소서 받기 -> 질문 생성 -> TTA오디오 생성 -> 오디오 db에 저장
# 질문에 대한 답변받기 -> STT변환 -> 답변에 대한 평가모델 -> 평가결과 db에 저장 -> PDF로 출력