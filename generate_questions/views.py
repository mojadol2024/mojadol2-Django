from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import os
import re

MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
state_dict = torch.load(MODEL_PATH + "/kobart_model.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")

class GenerateQuestionsView(APIView):
    def post(self, request):
        cover_letter = request.data.get("coverLetter")
        voucher_type = request.data.get("voucher")
        if not cover_letter:
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

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=296,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=1.2,
                    no_repeat_ngram_size=3,     
                )

            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            question_list = re.split(r'(?<=[.?])\s*', decoded)
            filtered_questions = []
            for q in question_list:
                q = q.strip()
                if len(q) > 8 and (q.endswith('.') or q.endswith('?')):
                    filtered_questions.append(q)
            return Response({"questions": filtered_questions}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
