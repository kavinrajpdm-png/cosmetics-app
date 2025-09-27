import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Cosmetics_cleaned dataset.xlsx")
    # Filter for moisturizers suitable for dry skin
    moisturizers_dry = df[(df['Label'] == "Moisturizer") & (df['Dry'] == 1)].reset_index(drop=True)
    return moisturizers_dry

df = load_data()

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
# Build Document-Term Matrix
# ---------------------------
M = len(corpus)
N = len(ingredient_idx)
A = np.zeros((M, N), dtype=int)

for i, tokens in enumerate(corpus):
    for t in tokens:
        j = ingredient_idx[t]
        A[i, j] = 1

# ---------------------------
# Apply t-SNE
# ---------------------------
tsne_model = TSNE(n_components=2, learning_rate=200, random_state=42)
tsne_features = tsne_model.fit_transform(A)
df['X'] = tsne_features[:, 0]
df['Y'] = tsne_features[:, 1]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Moisturizer Ingredient Similarity Explorer")
st.write("Select a moisturizer to view details and explore similar products:")

# Dropdown for selecting a product
product_name = st.selectbox("Choose a product:", df['Name'])
selected = df[df['Name'] == product_name]

st.subheader("Product Details")
st.write(selected[['Brand', 'Price', 'Rank', 'Ingredients']])

# ---------------------------
# Bokeh t-SNE Scatter Plot
# ---------------------------
source = ColumnDataSource(df)

p = figure(title="t-SNE of Moisturizers (Dry Skin)",
           x_axis_label="T-SNE 1", y_axis_label="T-SNE 2",
           width=800, height=600)
p.circle('X', 'Y', size=8, source=source, alpha=0.7)

hover = HoverTool(tooltips=[("Item", "@Name"),
                            ("Brand","@Brand"),
                            ("Price","$@Price"),
                            ("Rank","@Rank")])
p.add_tools(hover)

st.subheader("Interactive Similarity Plot")
st.bokeh_chart(p)

