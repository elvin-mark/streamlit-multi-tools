import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import numpy as np

st.set_page_config(page_title="Transformer Attention Visualizer", layout="wide")

st.title("🧠 Transformer Attention Mechanism Visualizer")
st.markdown(
    """
Visualize self-attention maps from a pre-trained **BERT** model.
"""
)


# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# User input
text = st.text_area(
    "✏️ Input Sentence", "The quick brown fox jumps over the lazy dog.", height=100
)

if text.strip():
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention = torch.stack(
        outputs.attentions
    )  # (layers, batch, heads, seq_len, seq_len)
    attention = attention.squeeze(1)  # Remove batch dim

    num_layers, num_heads, seq_len, _ = attention.shape

    # Layer and head selectors
    layer = st.slider("🔁 Select Layer", 0, num_layers - 1, 0)
    head = st.slider("🧠 Select Head", 0, num_heads - 1, 0)

    attn = attention[layer, head].detach().numpy()

    # Plot attention matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        square=True,
        cbar=True,
        ax=ax,
    )
    plt.title(f"Attention Map — Layer {layer}, Head {head}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Show token list
    with st.expander("🔍 Token List"):
        st.write(tokens)
