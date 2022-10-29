# Network-Analysis-of-BSS-in-Seoul-and-Daejeon

COVID-19 전후 서울 및 대전 공유자전거 이용패턴 변화 EDA 및 네트워크 중심성 변화 확인

이용 패턴 및 EDA (Data_Preprocessing&EDA_city.ipynb)


COVID-19 기간을 확진자 및 코로나 단계를 고려하여 구간을 나눔  
ex) 서울
![initial](https://user-images.githubusercontent.com/72389445/198822000-9baf2387-3a77-477d-a4f5-be8601f43879.png)

start_date|end_date
|---|---|
2020-02-19|2020-05-05
2020-05-06|2020-08-15
2020-08-16|2020-10-11
2020-10-12|2020-11-23
2020-11-24|2021-02-14
2021-02-15|2021-07-07
2021-07-08|2021-10-31
2021-11-01|2022-03-01
2022-03-02|2022-05-31

이후, 구간별로 네트워크를 계산한 뒤, 중심성 계산  (network centralities.py)  
2019년의 중심성과 COVID-19 기간의 중심성 차이를 비교하여 시각화 (Network_Visualization.ipynb)
