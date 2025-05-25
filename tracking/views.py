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


@method_decorator(csrf_exempt, name='dispatch')
class TrackingView(APIView):
    parser_classes = [JSONParser]
    def post(self, request):
        try:
            filename = request.data.get('filename')
            content_type = request.data.get('contentType')
            base64_data = request.data.get('fileData')

            if not base64_data:
                return Response({"error": "No file data provided"}, status=400)

            # base64 디코딩
            file_bytes = base64.b64decode(base64_data)

            temp_path = f"/tmp/{filename}"
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return Response({"error": "Failed to open video file"}, status=400)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30
            frame_interval = int(fps / 10)

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
                    break

                if frame_count % frame_interval == 0:
                    try:
                        gaze = GazeTracking()
                        gaze.refresh(frame)

                        is_blinking = gaze.is_blinking()
                        is_center = gaze.is_center()
                        is_left = gaze.is_left()
                        is_right = gaze.is_right()

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
                        print(f"Error processing frame {frame_count}: {str(e)}")
                        print(traceback.format_exc())

                frame_count += 1

            cap.release()

            if results["count"] == 0:
                return Response({"error": "No valid frames in video"}, status=400)

            average_score = results["score_sum"] / results["count"]

            return Response({
                "score": round(average_score, 2)
            })

        except Exception as e:
            print(f"Fatal error: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": "Internal server error"}, status=500)
