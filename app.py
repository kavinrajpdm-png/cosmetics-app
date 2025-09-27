import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import os

st.title("Moisturizer Ingredient Explorer")

# ---------------------------
# Check files in repo (debugging)
# ---------------------------
st.write("Files available in repo:", os.listdir())

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    # Make sure the Excel file name matches exactly
    df = pd.read_excel("cosmeticdata.xlsx")
    # Filter for moisturizers suitable for dry skin
    moisturizers_dry = df[(df['Label'] == "Moisturizer") & (df['Dry'] == 1)].reset_index(drop=True)
    return moisturizers_dry

try:
    df = load_data()
except FileNotFoundError:
    st.error("The dataset file 'cosmeticdata.xlsx' was not found. Make sure it is in the repo with app.py.")
    st.stop()

# ---------------------------
# Tokenize ingredients
# ---------------------------
ingredient_idx = {}
idx = 0
corpus = []

for ingredients_text in df['Ingredients']:
    if pd.isna(ingredients_text):
        tokens = []
    else:
        tokens = [t.strip().lower() for t in ingredients_text.split(',')]
    corpus.append(tokens)
    for t in tokens:
        if t not in ingredient_idx:
            ingredient_idx[t] = idx
            idx += 1

# ---------------------------
# Document-Term Matrix
# ---------------------------
M = len(corpus)
N = len(ingredient_idx)
A = np.zeros((M, N), dtype=int)

for i, tokens in enumerate(corpus):
    for t in tokens:
        j = ingredient_idx[t]
        A[i, j] = 1

# ---------------------------
# t-SNE dimensionality reduction
# ---------------------------
tsne_model = TSNE(n_components=2, learning_rate=200, random_state=42)
tsne_features = tsne_model.fit_transform(A)
df['X'] = tsne_features[:, 0]
df['Y'] = tsne_features[:, 1]

# ---------------------------
# Streamlit UI
# ---------------------------
product_name = st.selectbox("Select a moisturizer:", df['Name'])
selected = df[df['Name'] == product_name]

st.subheader("Product Details")
st.write(selected[['Brand', 'Price', 'Rank', 'Ingredients']])

# ---------------------------
# Bokeh t-SNE plot
# ---------------------------
source = ColumnDataSource(df)
p = figure(title="t-SNE of Moisturizers (Dry Skin)", width=800, height=600,
           x_axis_label="T-SNE 1", y_axis_label="T-SNE 2")
p.circle('X', 'Y', size=8, source=source, alpha=0.7)

hover = HoverTool(tooltips=[("Item", "@Name"),
                            ("Brand", "@Brand"),
                            ("Price", "$@Price"),
                            ("Rank", "@Rank")])
p.add_tools(hover)

st.subheader("Interactive Similarity Plot")
st.bokeh_chart(p)


