from django.shortcuts import render
from gaze_tracking import GazeTracking
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import numpy as np
import traceback
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import base64
import logging

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class TrackingView(APIView):
    parser_classes = [JSONParser]
    def post(self, request):
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

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                logger.error("비디오 파일 열기 실패")
                return Response({"error": "Failed to open video file"}, status=400)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30
            frame_interval = int(fps / 10)
            logger.debug(f"FPS: {fps}, 프레임 간격: {frame_interval}")

            results = {
                "blinking": 0,
                "center": 0,
                "left": 0,
                "right": 0,
                "no_face": 0,
                "off": 0,
                "score_sum": 0.0,
                "count": 0
            }

            frame_count = 0
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    logger.debug(f"프레임 읽기 실패 또는 끝: frame_count={frame_count}")
                    break

                if frame_count % frame_interval == 0:
                    try:
                        gaze = GazeTracking()
                        gaze.refresh(frame)

                        is_blinking = gaze.is_blinking()
                        is_center = gaze.is_center()
                        is_left = gaze.is_left()
                        is_right = gaze.is_right()

                        logger.debug(f"frame {frame_count}: blinking={is_blinking}, center={is_center}, left={is_left}, right={is_right}")

                        if not any([is_center, is_left, is_right]):
                            results["no_face"] += 1
                            score = 50.0
                        elif is_blinking:
                            results["blinking"] += 1
                            score = 60.0
                        elif is_center:
                            results["center"] += 1
                            score = 95.0
                        else:
                            results["off"] += 1
                            score = 70.0

                        results["score_sum"] += score
                        results["count"] += 1
                    except Exception as e:
                        logger.exception(f"프레임 {frame_count} 처리 중 오류: {str(e)}")

                frame_count += 1

            cap.release()

            if results["count"] == 0:
                logger.warning("유효한 프레임 없음")
                return Response({"error": "No valid frames in video"}, status=400)

            average_score = results["score_sum"] / results["count"]
            logger.info(f"평균 점수: {average_score}")

            return Response({
                "score": round(average_score, 2)
            })

        except Exception as e:
            logger.exception(f"치명적 오류 발생: {str(e)}")
            return Response({"error": "Internal server error"}, status=500)
