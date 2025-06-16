import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
tqdm.pandas()

#model_name = "Helsinki-NLP/opus-mt-en-vi"
model_name = "vinai/vinai-translate-en2vi-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_en_to_vi(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    print("번역 문장 : ", text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 엑셀 파일 불러오기
#df = pd.read_excel("D:\\temp\\번역.xlsx")
# D 컬럼 추출 및 번역
#df["Vietnamese_Translation"] = df.iloc[:, 3].astype(str).progress_apply(translate_en_to_vi)
# 결과 저장
#df.to_excel("D:\\temp\\번역_베트남어_완성본.xlsx", index=False)
#print("✅ 번역 완료! '번역결과파일.xlsx' 파일이 생성되었습니다.")

input = "No permission to restore"
out = translate_en_to_vi(input)
print("번역 결과:",out)

#inputs = tok("How are you today?", return_tensors="pt", padding=True)
#outs = model.generate(**inputs, num_beams=5)
#print("번역 결과:",tok.decode(outs[0], skip_special_tokens=True))