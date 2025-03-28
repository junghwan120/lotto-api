import logging
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random
import pandas as pd
from collections import Counter
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import schedule
import time
import threading
import requests

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# FastAPI 애플리케이션 생성
app = FastAPI()

# 현재 작업 디렉토리 (디버깅용)
logging.debug("Current working directory: " + os.getcwd())

# CSV 파일 경로 (프로젝트 폴더 내에 "로또당첨번호.csv" 있어야 함)
file_path = "로또당첨번호.csv"

# CSV 파일 읽기 및 전처리
try:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    logging.debug("CSV 파일 읽기 성공. DataFrame shape: %s", df.shape)
except Exception as e:
    logging.error("CSV 파일 읽기 실패: %s", e)
    df = None

if df is not None:
    # 필요 없는 컬럼 제거 및 날짜형 변환
    if "당첨금액" in df.columns:
        df = df.drop(columns=["당첨금액"])
    if "추첨일" in df.columns:
        df["추첨일"] = pd.to_datetime(df["추첨일"], errors="coerce")
    # 메인 번호 컬럼: "1", "2", "3", "4", "5", "6"
    main_number_cols = ["1", "2", "3", "4", "5", "6"]
    logging.debug("DataFrame columns: %s", df.columns.tolist())
    try:
        all_numbers = df[main_number_cols].values.flatten()
        number_counts = Counter(all_numbers)
    except Exception as e:
        logging.error("메인 번호 처리 오류: %s", e)
        number_counts = Counter()
else:
    logging.error("DataFrame이 None입니다.")
    number_counts = Counter()

# 통계 기반 추천 함수
def statistics_recommendation():
    try:
        top_numbers = [int(num) for num, count in number_counts.most_common(6)]
        top_numbers.sort()
        return top_numbers
    except Exception as e:
        logging.error("통계 추천 함수 오류: %s", e)
        return []

# 머신러닝 기반 추천: '번호1' 예측 (예시)
df_numbers = df[main_number_cols]
X = df_numbers.shift(1).dropna()
y = df_numbers.iloc[1:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ml_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)
latest_numbers = df_numbers.iloc[-1].values.reshape(1, -1)
predicted_number = int(np.round(ml_model.predict(latest_numbers)[0]))
logging.debug("머신러닝 기반 예측 번호 (번호1): %s", predicted_number)

def machine_learning_recommendation():
    try:
        rec = statistics_recommendation()
        rec[0] = predicted_number  # 통계 추천 결과의 첫 번째 번호를 예측 번호로 교체
        rec.sort()
        return rec
    except Exception as e:
        logging.error("머신러닝 추천 함수 오류: %s", e)
        return []

# FastAPI 엔드포인트
@app.get("/")
def read_root():
    try:
        return JSONResponse(content={"message": "로또 번호 추천 API에 오신 것을 환영합니다!"})
    except Exception as e:
        logging.error("루트 엔드포인트 오류: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/statistics")
def get_statistics_recommendation():
    try:
        rec = statistics_recommendation()
        return JSONResponse(content={"statistics_recommendation": rec})
    except Exception as e:
        logging.error("통계 엔드포인트 오류: %s", e)
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

@app.get("/lotto")
def get_lotto_numbers():
    try:
        numbers = random.sample(range(1, 46), 6)
        numbers.sort()
        return JSONResponse(content={"lotto_numbers": numbers})
    except Exception as e:
        logging.error("랜덤 엔드포인트 오류: %s", e)
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

@app.get("/ml")
def get_ml_recommendation():
    try:
        rec = machine_learning_recommendation()
        return JSONResponse(content={"ml_recommendation": rec})
    except Exception as e:
        logging.error("머신러닝 엔드포인트 오류: %s", e)
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

# 자동 스케줄링: 매 분마다 job() 실행 (테스트용; 실제 운영 시에는 매주 금요일 23:00 등으로 변경)
def integrated_recommendation(n_trials=50):
    # 예시: 통계 기반 추천 결과를 그대로 반환 (추후 통합 추천 로직으로 확장 가능)
    return statistics_recommendation()

def job():
    final_numbers = integrated_recommendation()
    message = "매주 추천 로또 번호: " + str(final_numbers)
    print(message)
    # 알림 전송: 카카오톡 API를 사용하여 메시지를 보냅니다.
    send_kakao_message(message)

def run_scheduler():
    # 테스트용: 매 1분마다 job() 실행
    schedule.every(1).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# 카카오톡 알림 전송 함수 (예시)
def send_kakao_message(message: str):
    try:
        # 아래 URL 및 payload는 카카오톡 메모 API 예시입니다.
        # 실제 사용 시 카카오 개발자 문서를 참고하여 앱 등록, Access Token 발급 등을 진행하세요.
        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        access_token = "YOUR_KAKAO_ACCESS_TOKEN"  # 실제 토큰으로 교체
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        # template_object는 JSON 문자열로 전송됩니다.
        template_object = {
            "object_type": "text",
            "text": message,
            "link": {
                "web_url": "https://developers.kakao.com",
                "mobile_web_url": "https://developers.kakao.com"
            },
            "button_title": "자세히 보기"
        }
        data = {
            "template_object": json.dumps(template_object, ensure_ascii=False)
        }
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            logging.info("Kakao 메시지 전송 성공")
        else:
            logging.error("Kakao 메시지 전송 실패, 상태 코드: %s, 응답: %s", response.status_code, response.text)
    except Exception as e:
        logging.error("Kakao 메시지 전송 오류: %s", e)

# 서버 실행 (배포 시, 아래 if __name__ == "__main__": 블록 사용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="debug")