from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import os
import re
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.debug(f"모델 로드 시작: MODEL_PATH={MODEL_PATH}, DEVICE={DEVICE}")
model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
state_dict = torch.load(MODEL_PATH + "/kobart_model.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
logger.info("모델 로드 완료")

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
logger.info("토크나이저 로드 완료")

class GenerateQuestionsView(APIView):
    def post(self, request):
        logger.debug("[GenerateQuestionsView] POST 요청 시작")
        cover_letter = request.data.get("coverLetter")
        voucher_type = request.data.get("voucher")
        logger.debug(f"받은 데이터: voucher_type={voucher_type}, cover_letter 길이={len(cover_letter) if cover_letter else 0}")

        if not cover_letter:
            logger.warning("coverLetter 필드 없음")
            return Response({"error": "coverLetter 필드는 필수입니다."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            inputs = tokenizer(
                cover_letter,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="max_length"
            )
            inputs["input_ids"] = inputs["input_ids"].to(DEVICE)
            inputs["attention_mask"] = inputs["attention_mask"].to(DEVICE)
            logger.debug("토크나이저 인풋 생성 완료")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=296,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=1.2,
                    no_repeat_ngram_size=3,     
                )
            logger.debug("모델 생성 완료")

            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.debug(f"디코딩 결과: {decoded}")

            question_list = re.split(r'(?<=[.?])\s*', decoded)
            filtered_questions = []
            for q in question_list:
                q = q.strip()
                if len(q) > 8 and (q.endswith('.') or q.endswith('?')):
                    filtered_questions.append(q)
            logger.info(f"생성된 질문 수: {len(filtered_questions)}")

            return Response({"questions": filtered_questions}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception(f"질문 생성 중 오류 발생: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
