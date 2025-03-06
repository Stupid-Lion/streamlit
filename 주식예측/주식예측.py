import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from matplotlib import rcParams, font_manager

# 한글 폰트 설정 (배포 환경에서 폰트 파일 경로 설정)
font_path = 'streamlit/주식예측/NanumGothic-Bold.ttf'  # 실제 경로로 수정해야 합니다.
font_prop = font_manager.FontProperties(fname=font_path)

# 한글 폰트 적용
rcParams['font.family'] = font_prop.get_name()
rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# 앱 제목
st.title("주식 예측 웹 애플리케이션")

# 인기 있는 주식 티커 목록 표로 표시
stock_data = {
    "회사": ["Apple", "Tesla", "Google", "Amazon", "Microsoft", "NVIDIA", "Meta (Facebook)", "Netflix", "AMD", "Disney"],
    "주식 티커": ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "NVDA", "META", "NFLX", "AMD", "DIS"]
}

# DataFrame으로 변환
stock_df = pd.DataFrame(stock_data)

# 주식 티커 목록을 표로 출력
st.write("인기 있는 주식 티커 목록:")
st.dataframe(stock_df)

# 사용자로부터 주식 티커 입력 받기
ticker = st.text_input("원하는 주식 티커를 입력하세요 (예: AAPL, TSLA)", "AAPL")

# 데이터 시작일과 종료일 설정
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

# 주식 데이터 가져오기
data = yf.download(ticker, start=start_date, end=end_date)

# 데이터 역순 정렬 (최근 날짜부터 보기)
data = data.sort_index(ascending=False)

# 데이터 확인
if st.checkbox('데이터 확인'):
    st.write(data)

# 종가 데이터를 사용하여 예측하기
X = np.array(range(len(data))).reshape(-1, 1)  # 날짜 순서
y = data['Close'].values  # 종가

# 선형 회귀 모델로 예측
model = LinearRegression()
model.fit(X, y)

# 예측값 생성
predictions = model.predict(X)

# 예측값 시각화
plt.figure(figsize=(10, 6))
plt.plot(data.index, y, label="실제 종가", color='blue')
plt.plot(data.index, predictions, label="예측 종가", color='red', linestyle='--')
plt.title(f"{ticker} 주식 예측")
plt.xlabel("날짜")
plt.ylabel("가격")
plt.legend()
st.pyplot(plt)

# 미래 예측: 다음 7일 예측
future_dates = np.array(range(len(data), len(data) + 7)).reshape(-1, 1)
future_predictions = model.predict(future_dates)

st.write(f"미래 7일 예측 (다음 7일의 주식 가격):")
for i in range(7):
    st.write(f"{(datetime.today() + timedelta(days=i+1)).strftime('%Y-%m-%d')}: {future_predictions[i].item():.2f} USD")
