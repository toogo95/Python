import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
tqdm.pandas()


tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")

################## CPU-based sequence translation ##################
def translate_en2vi(en_text: str) -> str:
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_text = " ".join(vi_text)
    return vi_text

# 번역_1000
# 100%|##########| 999/999 [14:31<00:00,  1.15it/s]
# [Done] exited with code=0 in 881.066 seconds (per 0.88sec)

# 번역_3561
# 14%|#4        | 500/3560 [08:54<33:31,  1.52it/s] (per 0.93sec)
# 28%|##8       | 999/3560 [14:21<32:38,  1.31it/s] (per 0.86sec)
# 56%|#####6    | 1999/3560 [28:18<17:14,  1.51it/s] (per 0.84sec)
# 84%|########4 | 2999/3560 [43:00<11:57,  1.28s/it] (per 0.86sec)
#100%|##########| 3560/3560 [55:54<00:00,  1.06it/s] (per 0.94sec)
# [Done] exited with code=0 in 3362.995 seconds (per 0.94sec)
###################################################################

################ GPU-based sequence translation ########################
# import torch
# device_en2vi = torch.device("cuda")
# model_en2vi.to(device_en2vi)

# def translate_en2vi(en_texts: str) -> str:
#     input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(device_en2vi)
#     output_ids = model_en2vi.generate(
#         **input_ids,
#         decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
#         num_return_sequences=1,
#         num_beams=5,
#         early_stopping=True
#     )
#     vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
#     return vi_texts
###################################################################

# en_text = "Essential information was not provided."
# print("번역 문장:",en_text)
# print("번역 결과:",translate_en2vi(en_text))

# 엑셀 파일 불러오기
filename = "번역_3561"
df = pd.read_excel("D:\\01.Work\\05.Study\\Translate_model\\traslate\\en2vi\\" + filename + ".xlsx")
# D 컬럼 추출 및 번역
df["Vietnamese_Translation"] = df.iloc[:, 3].astype(str).progress_apply(translate_en2vi)
# 결과 저장
df.to_excel("D:\\01.Work\\05.Study\\Translate_model\\traslate\\en2vi\\" + filename + "_번역완료.xlsx", index=False)
print("✅ 번역 완료! " + filename + "_번역완료.xlsx 파일이 생성되었습니다.")