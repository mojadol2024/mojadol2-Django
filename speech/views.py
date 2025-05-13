from django.shortcuts import render
import whisper
import tempfile
import ffmpeg
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import fasttext

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
            model = whisper.load_model("large-v3")
            result = model.transcribe(temp_audio_path, language="ko")

            return Response({"text": result["text"]}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TTSView(APIView):
    def post(self, request):
        """
        @inproceedings{
            kargaran2023glotlid,
            title={{GlotLID}: Language Identification for Low-Resource Languages},
            author={Kargaran, Amir Hossein and Imani, Ayyoob and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
            booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
            year={2023},
            url={https://openreview.net/forum?id=dl4e3EBz5j}
            }
        """
        model = fasttext.load_model("../model/tts.bin")
        model.predict(request.data["text"])
        
        