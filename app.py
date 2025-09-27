import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

st.set_page_config(page_title="Cosmetic Product Explorer", layout="wide")
st.title("Cosmetic Product Ingredient Explorer")

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
# Select product category
# ---------------------------
category = st.selectbox("Select Product Category:", df['Label'].unique())
category_df = df[df['Label'] == category].reset_index(drop=True)

if category_df.empty:
    st.error(f"No products found for category: {category}")
    st.stop()

# ---------------------------
# Select skin type
# ---------------------------
skin_types = ['All', 'Dry', 'Oily', 'Normal', 'Combination', 'Sensitive']
if 'Dry' in category_df.columns and 'Oily' in category_df.columns and 'Normal' in category_df.columns and 'Combination' in category_df.columns and 'Sensitive' in category_df.columns:
    skin_filter = st.selectbox("Filter by Skin Type:", skin_types)
    if skin_filter != "All":
        category_df = category_df[category_df[skin_filter] == 1].reset_index(drop=True)
        if category_df.empty:
            st.error(f"No products found for {skin_filter} skin in this category")
            st.stop()

# ---------------------------
# Tokenize ingredients and create document-term matrix
# ---------------------------
ingredient_idx = {}
idx = 0
corpus = []

for ingredients_text in category_df['Ingredients']:
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
# Compute t-SNE
# ---------------------------
if A.shape[0] > 1 and A.shape[1] > 0:
    tsne_model = TSNE(n_components=2, learning_rate=200, random_state=42)
    tsne_features = tsne_model.fit_transform(A)
    category_df['X'] = tsne_features[:, 0]
    category_df['Y'] = tsne_features[:, 1]
else:
    category_df['X'] = np.arange(len(category_df))
    category_df['Y'] = np.arange(len(category_df))

# ---------------------------
# Product selection
# ---------------------------
product_name = st.selectbox("Select a product:", category_df['Name'])
selected = category_df[category_df['Name'] == product_name]

st.subheader("Product Details")
st.write(selected[['Brand', 'Price', 'Rank', 'Ingredients']])

# ---------------------------
# Interactive t-SNE Plot with Plotly
# ---------------------------
fig = px.scatter(
    category_df,
    x='X',
    y='Y',
    hover_data=['Name', 'Brand', 'Price', 'Rank'],
    color='Brand',
    title=f"{category} Ingredient Similarity (t-SNE) - Skin Type: {skin_filter}"
)

st.subheader("Interactive Ingredient Similarity Plot")
st.plotly_chart(fig, use_container_width=True)
