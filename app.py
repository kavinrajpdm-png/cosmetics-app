import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

st.set_page_config(page_title="Moisturizer Ingredient Explorer", layout="wide")
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
# Tokenize ingredients and create document-term matrix
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

M = len(corpus)
N = len(ingredient_idx)
A = np.zeros((M, N), dtype=int)

for i, tokens in enumerate(corpus):
    for t in tokens:
        j = ingredient_idx[t]
        A[i, j] = 1

# ---------------------------
# Compute t-SNE (safe)
# ---------------------------
if A.shape[0] > 1 and A.shape[1] > 0:
    tsne_model = TSNE(n_components=2, learning_rate=200, random_state=42)
    tsne_features = tsne_model.fit_transform(A)
    moisturizers_dry['X'] = tsne_features[:, 0]
    moisturizers_dry['Y'] = tsne_features[:, 1]
else:
    # fallback for very small dataset
    moisturizers_dry['X'] = np.arange(len(moisturizers_dry))
    moisturizers_dry['Y'] = np.arange(len(moisturizers_dry))

# ---------------------------
# Product selection
# ---------------------------
product_name = st.selectbox("Select a moisturizer:", moisturizers_dry['Name'])
selected = moisturizers_dry[moisturizers_dry['Name'] == product_name]

st.subheader("Product Details")
st.write(selected[['Brand', 'Price', 'Rank', 'Ingredients']])

# ---------------------------
# Interactive t-SNE Plot with Plotly
# ---------------------------
fig = px.scatter(
    moisturizers_dry,
    x='X',
    y='Y',
    hover_data=['Name', 'Brand', 'Price', 'Rank'],
    color='Brand',
    title="Moisturizer Ingredient Similarity (t-SNE)"
)

st.subheader("Interactive Ingredient Similarity Plot")
st.plotly_chart(fig, use_container_width=True)
