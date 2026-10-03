import pandas as pd
import streamlit as st
from modelcompare import compare

st.set_page_config(page_title="modelcompare", page_icon="🏆", layout="centered")
st.title("🏆 modelcompare")
st.write("Upload a CSV, pick the column to predict, and compare 5 ML models with cross-validation.")

file = st.file_uploader("CSV file", type="csv")
if file:
    df = pd.read_csv(file)
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    target = st.selectbox("Column to predict", df.columns, index=len(df.columns) - 1)
    folds = st.slider("Cross-validation folds", 3, 10, 5)

    if st.button("Compare models", type="primary"):
        with st.spinner("Training models..."):
            try:
                res = compare(df, target, folds)
            except Exception as e:
                st.error(f"Could not compare models: {e}")
                st.stop()
        metric = res.columns[1]
        st.subheader(f"Task: {res.attrs.get('task', 'unknown')} ({metric})")
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.bar_chart(res.set_index("model")[metric])
        st.success(f"🏆 Best model: {res.loc[0, 'model']}")
else:
    st.info("Upload a CSV to get started.")