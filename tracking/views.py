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
import aiohttp
import asyncio
import json
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from django.core.management.base import BaseCommand
import os

logger = logging.getLogger(__name__)

async def download_video(url, filename):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                temp_path = f"/tmp/{filename}"
                with open(temp_path, "wb") as f:
                    f.write(await resp.read())
                logger.info(f"다운로드 완료: {temp_path}")
                return temp_path
            else:
                logger.error(f"다운로드 실패: {resp.status}")
                return None

async def process_video(video_url, filename, interviewId):
    video_path = await download_video(video_url, filename)
    if not video_path:
        logger.error("비디오 다운로드 실패")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("비디오 열기 실패")
        os.remove(video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    frame_interval = int(fps / 10)
    logger.info(f"FPS: {fps}, 프레임 간격: {frame_interval}")

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
                logger.exception(f"프레임 {frame_count} 처리 중 오류: {e}")

        frame_count += 1

    cap.release()
    os.remove(video_path)
    logger.info(f"파일 삭제 완료: {video_path}")

    if results["count"] == 0:
        logger.warning("유효한 프레임 없음")
        return None

    average_score = results["score_sum"] / results["count"]
    logger.info(f"평균 점수: {average_score}")
    
    score = round(average_score, 2)
    
    await send_result_to_kafka(score, interviewId)
    return score


async def send_result_to_kafka(score, interviewId):
    producer = AIOKafkaProducer(bootstrap_servers='pbl2-kafka1:29092,pbl2-kafka2:29093,pbl2-kafka3:29094')
    await producer.start()
    try:
        result_data = {
            'score': score,
            'interviewId':interviewId
        }
        await producer.send_and_wait('interview-video-result', json.dumps(result_data).encode('utf-8'))
        logger.info(f"결과 전송 성공: {result_data}")
    finally:
        await producer.stop()

class Command(BaseCommand):
    async def consume(self):
        consumer = AIOKafkaConsumer(
            'interview-video',
            bootstrap_servers='pbl2-kafka1:29092,pbl2-kafka2:29093,pbl2-kafka3:29094',
            group_id='django-consumer-group'
        )
        await consumer.start()
        try:
            async for msg in consumer:
                try:
                    data = json.loads(msg.value.decode('utf-8'))
                    video_url = data.get('videoUrl')
                    filename = data.get('filename')
                    interviewId = data.get('interviewId')
                    if not video_url or not filename:
                        logger.error(f"메시지 데이터 불완전: {data}")
                        continue

                    logger.info(f"Kafka 메시지 수신: videoUrl={video_url}, filename={filename}")
                    score = await process_video(video_url, filename, interviewId)
                    if score is not None:
                        logger.info(f"처리 성공: 평균 점수 = {score}")
                        # 여기서 DB 저장이나 다른 로직 추가 가능
                    else:
                        logger.warning("비디오 처리 실패")
                except Exception as e:
                    logger.exception(f"Kafka 메시지 처리 중 오류: {e}")
        finally:
            await consumer.stop()
            logger.info("Kafka consumer 종료")

    def handle(self, *args, **kwargs):
        asyncio.run(self.consume())
        
        
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
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"임시 파일 삭제됨: {temp_path}")