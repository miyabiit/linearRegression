import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scikit-learn.datasets import load_boston
from scikit-learn.model_selection import train_test_split

st.title("ボストン市の住宅価格の重回帰分析")
