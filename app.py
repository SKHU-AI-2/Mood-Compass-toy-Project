from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel  # BaseModel 임포트 추가
import torch
from transformers import AutoTokenizer, BertForSequenceClassification


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=6)
model.classifier = torch.nn.Linear(model.config.hidden_size, 6)  # 커스텀 분류기 재정의

# 저장된 모델 가중치 로드
model_path = "bert_6_emotions.pt"  # 모델 가중치 파일 경로를 실제 경로로 변경해야 함
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
model.eval()  # 평가 모드로 설정

# 감정 ID와 이름 매핑
emotion_mapping = {
    0: "기쁨",
    1: "당황",
    2: "분노",
    3: "불안",
    4: "상처",
    5: "슬픔"
}

class TextItem(BaseModel):
    text: str

@app.get("/")
async def read_root(request: Request):
    # 메인 페이지를 렌더링
    return templates.TemplateResponse("diary.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, text: str = Form(...)):
    # 입력된 텍스트를 모델로 분석
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_emotion = emotion_mapping[predicted_class_id]
    
    # 분석 결과를 보여주는 페이지로 리디렉션
    return templates.TemplateResponse("result.html", {"request": request, "emotion": predicted_emotion})