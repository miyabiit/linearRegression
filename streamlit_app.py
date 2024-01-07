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




