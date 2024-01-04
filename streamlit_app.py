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

if st.checkbox("show data"):
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
