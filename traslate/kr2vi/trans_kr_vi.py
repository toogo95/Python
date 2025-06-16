import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "ofu-ai/mbart-large-50-mmt-ko-vi"

# 모델과 토크나이저 로드 (최초 실행 시 다운로드, 이후 캐시에서 로드됨)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ko_KR")
model = MBartForConditionalGeneration.from_pretrained(model_name)

# 예시 입력
text = "오늘 날씨가 어때요?"
inputs = tokenizer(text, return_tensors="pt")

# 베트남어로 강제 번역
generated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"])

# 출력
print("번역 결과:", tokenizer.decode(generated[0], skip_special_tokens=True))