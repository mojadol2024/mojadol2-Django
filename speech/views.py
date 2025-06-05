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
from rest_framework.parsers import JSONParser
import logging
import base64
logger = logging.getLogger(__name__)

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
    parser_classes = [JSONParser]

    def post(self, request):
        temp_path = None
        try:
            logger.debug("POST 요청 시작")

            filename = request.data.get('filename')
            content_type = request.data.get('contentType')
            base64_data = request.data.get('fileData')
            logger.debug(f"받은 데이터: filename={filename}, content_type={content_type}")

            if not base64_data:
                logger.warning("fileData 없음")
                return Response({"error": "No file data provided"}, status=400)

            file_bytes = base64.b64decode(base64_data)
            temp_path = f"/tmp/{filename}"
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            logger.debug(f"파일 저장 완료: {temp_path}")

            # whisper로 STT 처리
            result = analyze_speaking_speed(temp_path)
            logger.debug(f"STT 처리 결과: {result}")

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception(f"STT 처리 중 예외 발생: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"임시 비디오 파일 삭제됨: {temp_path}")

# 자소서 받기 -> 질문 생성 -> TTA오디오 생성 -> 오디오 db에 저장
# 질문에 대한 답변받기 -> STT변환 -> 답변에 대한 평가모델 -> 평가결과 db에 저장 -> PDF로 출력