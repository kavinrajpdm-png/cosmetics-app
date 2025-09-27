import streamlit as st
import pandas as pd
import numpy as np

st.title("Moisturizer Ingredient Explorer")

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Cosmeticdata.xlsx")
        return df
    except FileNotFoundError:
        st.error("Dataset 'Cosmeticdata.xlsx' not found in repo!")
        st.stop()

df = load_data()

# ---------------------------
# Filter moisturizers for dry skin
# ---------------------------
moisturizers_dry = df[(df['Label'] == "Moisturizer") & (df['Dry'] == 1)].reset_index(drop=True)

if moisturizers_dry.empty:
    st.error("No moisturizers for dry skin found!")
    st.stop()

# ---------------------------
# Product selection
# ---------------------------
product_name = st.selectbox("Select a moisturizer:", moisturizers_dry['Name'])
selected = moisturizers_dry[moisturizers_dry['Name'] == product_name]

st.subheader("Product Details")
st.write(selected[['Brand', 'Price', 'Rank', 'Ingredients']])

# ---------------------------
# Safe scatter plot using Streamlit
# ---------------------------
st.subheader("Interactive Plot (Random Coordinates)")

# Generate safe numeric X and Y
moisturizers_dry['X'] = np.arange(len(moisturizers_dry))
moisturizers_dry['Y'] = np.arange(len(moisturizers_dry))

# Plot using Streamlit
st.scatter_chart(moisturizers_dry[['X', 'Y']])
