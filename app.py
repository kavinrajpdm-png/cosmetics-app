import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

st.title("Moisturizer Ingredient Explorer")

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Cosmeticdata.xlsx")  # Ensure file name matches
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
# Streamlit UI - product selection
# ---------------------------
product_name = st.selectbox("Select a moisturizer:", moisturizers_dry['Name'])
selected = moisturizers_dry[moisturizers_dry['Name'] == product_name]

st.subheader("Product Details")
st.write(selected[['Brand', 'Price', 'Rank', 'Ingredients']])

# ---------------------------
# Safe Bokeh plot - ensure numeric coordinates
# ---------------------------
# Create safe numeric X and Y columns
moisturizers_dry = moisturizers_dry.copy()
moisturizers_dry['X'] = np.arange(len(moisturizers_dry), dtype=float)
moisturizers_dry['Y'] = np.arange(len(moisturizers_dry), dtype=float)

# Remove NaNs in columns required for hover
for col in ['Name', 'Brand', 'Price', 'Rank']:
    if col not in moisturizers_dry.columns:
        moisturizers_dry[col] = "Unknown"
    else:
        moisturizers_dry[col] = moisturizers_dry[col].fillna("Unknown")

source = ColumnDataSource(moisturizers_dry)

p = figure(title="Moisturizer Similarity Plot (Safe Coordinates)", width=800, height=600,
           x_axis_label="X", y_axis_label="Y")
p.circle('X', 'Y', size=8, source=source, alpha=0.7)

hover = HoverTool(tooltips=[("Item", "@Name"), ("Brand", "@Brand"), ("Price", "@Price"), ("Rank", "@Rank")])
p.add_tools(hover)

st.subheader("Interactive Plot")
st.bokeh_chart(p)
