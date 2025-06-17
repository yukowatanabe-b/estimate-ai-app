# estimate_app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# データ読み込み
df = pd.read_csv("estimate_data.csv", encoding="cp932")

# ラベルエンコーディング（カテゴリ変数を数値に変換）
le_material = LabelEncoder()
le_work = LabelEncoder()
df["材料種別"] = le_material.fit_transform(df["材料種別"])
df["工種"] = le_work.fit_transform(df["工種"])
df = df.dropna(subset=["見積金額"])

# 学習用データ
X = df[["面積", "材料種別", "工種"]]
y = df["見積金額"]

# モデル学習
model = RandomForestRegressor()
model.fit(X, y)

# Streamlit UI
st.title("AI見積金額予測アプリ")

# 入力欄
area = st.number_input("面積（㎡）", min_value=1)
material = st.selectbox("材料種別", le_material.classes_)
work = st.selectbox("工種", le_work.classes_)

# 予測実行
if st.button("予測する"):
    mat_encoded = le_material.transform([material])[0]
    work_encoded = le_work.transform([work])[0]
    input_data = pd.DataFrame([[area, mat_encoded, work_encoded]],
                              columns=["面積", "材料種別", "工種"])
    prediction = model.predict(input_data)[0]
    st.success(f"予測見積金額：¥{int(prediction):,}")
