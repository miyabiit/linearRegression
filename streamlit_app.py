import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
#from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set()
st.title("California 住宅価格の重回帰分析")

# load dataset
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
df["PRICES"] = dataset.target

if st.checkbox("Show data"):
    df

if st.checkbox("Description"):
    dataset.DESCR

# visualization
if st.checkbox("Visualization"):
    checked_variable = st.selectbox("please one variable:",df.drop(columns="PRICES").columns)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.scatter(x=df[checked_variable], y=df["PRICES"], s=1)
    plt.xlabel(checked_variable)
    plt.ylabel("PRICES")
    st.pyplot(fig)

"""
## 前処理
"""
# 学習時に使用しない変数を取り除く
features_notUsed = st.multiselect("使用しない変数を選択してください", 
                                  df.drop(columns = "PRICES").columns)
df = df.drop(columns = features_notUsed)

# 対数変換を行う
left_column, right_column = st.columns([1,2])
bool_log = left_column.radio("対数変換を行いますか？", ("Yes", "No"))
df_log, log_features = df.copy(), []
if bool_log == "Yes":
    log_features = right_column.multiselect(
            "対数変換を適用する目的変数、説明変数を選択してください",
            df.columns)
    df_log[log_features] = np.log(df_log[log_features]) 
                                  
# standard
left_column, right_column = st.columns([1,2])
bool_std = left_column.radio("Standard?", ("Yes", "No"))
df_std = df_log.copy()
if bool_std == "Yes":
    std_features_notUsed = right_column.multiselect(
            "標準化を適用しない変数を選択してください（例えば、質的変数)",
            df_log.drop(columns=["PRICES"]).columns)
    std_features_chosen = []
    for name in df_log.drop(columns=["PRICES"]).columns:
        if name in std_features_notUsed:
            continue
        else:
            std_features_chosen.append(name)
    sscaler = preprocessing.StandardScaler()
    sscaler.fit(df_std[std_features_chosen])
    df_std[std_features_chosen] = sscaler.transform(df_std[std_features_chosen])

"""
## データセットを訓練用と検証用に分割
"""

left_column, right_column = st.columns([1,2])
test_size = left_column.number_input(
        "検証用のデータサイズ（1.0 - 0.0):",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.2,
        step = 0.1
        )
random_seed = right_column.number_input(
        "ランダムシードの設定(0以上):",
        min_value = 0,
        value = 0,
        step = 1,
        )
# データセットの分割
x_train, x_value, y_train, y_value = train_test_split(
        df_std.drop(columns=["PRICES"]),
        df_std["PRICES"],
        test_size=test_size,
        random_state=random_seed
        )

# 回帰モデルの作成と訓練
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred_train = regressor.predict(x_train)
y_pred_value = regressor.predict(x_value)

if "PRICES" in log_features:
    y_pred_value, y_pred_train = np.exp(y_pred_value), np.exp(y_pred_train)
    y_value, y_train = np.exp(y_value), np.exp(y_train)

"""
## 結果の表示
"""

# Coefficient
if st.checkbox("Coefficient"):
    coefficient = regressor.coef_
    df_coef = pd.DataFrame(
            coefficient,
            columns=["Coefficient"],
            index=df_std.drop(columns=["PRICES"]).columns)
    df_coef


# モデルの精度
R2 = r2_score(y_value, y_pred_value)
st.write(f'R2値:{R2:.2f}')

"""
## 表示
"""
left_column, right_column = st.columns([1,2])
show_train = left_column.radio(
        "訓練データの結果を表示:",
        ("Yes","No"))
show_value = right_column.radio(
        "検証データの結果を表示:",
        ('Yes','No'))

# 目的変数の最大値を取得
y_max_train = max([max(y_train),max(y_pred_train)])
y_max_value = max([max(y_value),max(y_pred_value)])
y_max = int(max([y_max_train,y_max_value]))

# 動的に軸の範囲を設定
left_column, right_column = st.columns([1,2])
x_min = left_column.number_input(
        'X軸の最小値',
        value=0, step=1)
x_max = right_column.number_input(
        'X軸の最大値',
        value=y_max, step=1)

left_column, right_column = st.columns([1,2])
y_min = left_column.number_input(
        'Y軸の最小値',
        value=0, step=1)
y_max = right_column.number_input(
        'Y軸の最大値',
        value=y_max, step=1)

# グラフ
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
if show_train == 'Yes':
    ax.scatter(y_train, y_pred_train, linewidths=0, color="r", s=2)
if show_value == 'Yes':
    ax.scatter(y_value, y_pred_value, linewidths=0, color="b", s=2)
ax.grid(True)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("PRICES")
ax.set_ylabel("Prediction of PRICES")
ax.legend(['TrainingData','ValidationData'])
st.pyplot(fig) 

# coefficient

