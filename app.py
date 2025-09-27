import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

st.set_page_config(page_title="Moisturizer Ingredient Explorer", layout="wide")
st.title("Moisturizer Ingredient Explorer")

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Cosmeticdata.xlsx")  # Make sure this file is in the same repo
        return df
    except FileNotFoundError:
        st.error("Dataset 'Cosmeticdata.xlsx' not found in the repo!")
        st.stop()

df = load_data()

# ---------------------------
# Filter moisturizers for dry skin
# ---------------------------
moisturizers_dry = df[(df['Label'] == "Moisturizer") & (df['Dry'] == 1)].reset_index(drop=True)

if moisturizers_dry.empty:
    st.error("No moisturizers for dry skin found in the dataset!")
    st.stop()

# ---------------------------
# Tokenize ingredients
# ---------------------------
ingredient_idx = {}
idx = 0
corpus = []

for ingredients_text in moisturizers_dry['Ingredients']:
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
if A.shape[0] > 1 and A.shape[1] > 0:  # ensure matrix is not empty
    tsne_model = TSNE(n_components=2, learning_rate=200, random_state=42)
    tsne_features = tsne_model.fit_transform(A)
    moisturizers_dry['X'] = tsne_features[:, 0]
    moisturizers_dry['Y'] = tsne_features[:, 1]
else:
    moisturizers_dry['X'] = 0
    moisturizers_dry['Y'] = 0

# ---------------------------
# Streamlit UI - product selection
# ---------------------------
product_name = st.selectbox("Select a moisturizer:", moisturizers_dry['Name'])
selected = moisturizers_dry[moisturizers_dry['Name'] == product_name]

st.subheader("Product Details")
st.write(selected[['Brand', 'Price', 'Rank', 'Ingredients']])

# ---------------------------
# Bokeh t-SNE plot
# ---------------------------
source = ColumnDataSource(moisturizers_dry)

p = figure(title="t-SNE Plot of Moisturizers (Dry Skin)",
           width=800, height=600,
           x_axis_label="T-SNE 1", y_axis_label="T-SNE 2")

p.circle('X', 'Y', size=8, source=source, alpha=0.7)

hover = HoverTool(tooltips=[("Item", "@Name"),
                            ("Brand", "@Brand"),
                            ("Price", "$@Price"),
                            ("Rank", "@Rank")])
p.add_tools(hover)

st.subheader("Interactive Similarity Plot")
st.bokeh_chart(p)
