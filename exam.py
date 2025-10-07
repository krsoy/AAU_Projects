import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Local CSV test", layout="wide")
st.title("✅ Local CSV quick test")

# 自动检测当前脚本路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "survey_results_public.csv")

st.write("CSV path:", CSV_PATH)

try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    st.success(f"Loaded successfully! Rows: {df.shape[0]}, Cols: {df.shape[1]}")
    st.dataframe(df.head(10))
except Exception as e:
    st.error(f"❌ Failed to read CSV: {e}")