from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# 기본 라우트 (홈)
@app.get("/")
def read_root():
    return JSONResponse(content={"message": "로또 번호 추천 API에 오신 것을 환영합니다!"}, ensure_ascii=False)

# 랜덤 추천 엔드포인트
@app.get("/lotto")
def get_lotto_numbers():
    numbers = random.sample(range(1, 46), 6)
    numbers.sort()
    return JSONResponse(content={"lotto_numbers": numbers}, ensure_ascii=False)

# 머신러닝 기반 추천 엔드포인트 (예시)
@app.get("/predict")
def predict_lotto_numbers():
    predicted = machine_learning_recommendation()
    return JSONResponse(content={"predicted_lotto_numbers": predicted}, ensure_ascii=False)

# 최종 통합 추천 엔드포인트
@app.get("/integrated")
def get_integrated_numbers():
    final_numbers = integrated_recommendation()
    return JSONResponse(content={"final_recommended_numbers": final_numbers}, ensure_ascii=False)