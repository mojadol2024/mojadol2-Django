from django.shortcuts import render
from gaze_tracking import GazeTracking
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import numpy as np


class TrackingView(APIView):
    def post(self, request):
        video_file = request.FILES.get("video")

        if not video_file:
            return Response({"error": "No video uploaded"}, status=400)

        temp_path = f"/tmp/{video_file.name}"
        with open(temp_path, "wb+") as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return Response({"error": "Failed to open video file"}, status=400)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # FPS 못 가져오면 기본값 30으로 가정
        frame_interval = int(fps / 10)  # 10 FPS로 줄이기 위한 간격

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

            frame_count += 1

        cap.release()

        if results["count"] == 0:
            return Response({"error": "No valid frames in video"}, status=400)

        average_score = results["score_sum"] / results["count"]

        return Response({
            "status_counts": {
                "center": results["center"],
                "left": results["left"],
                "right": results["right"],
                "blinking": results["blinking"],
                "off": results["off"],
                "no_face": results["no_face"]
            },
            "average_score": round(average_score, 2),
            "total_frames": results["count"]
        })

