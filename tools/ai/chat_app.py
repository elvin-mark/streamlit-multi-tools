import streamlit as st
import openai

# --- Configure OpenAI client for local LLM ---
client = openai.OpenAI(
    base_url="http://localhost:8080/v1", api_key="sk-no-key-required"
)

st.set_page_config(page_title="LLM Chat", layout="wide")

# --- Title ---
st.title("🧠 Chat with Local LLM")

# --- Chat history session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
        }
    ]

# --- Display chat history ---
for msg in st.session_state.messages[1:]:  # skip system message in UI
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
prompt = st.chat_input("Ask me anything...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)

    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Create a placeholder for assistant message
        assistant_placeholder = st.chat_message("assistant").empty()
        streamed_text = ""

        # Send streaming request to LLM
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            stream=True,
        )

        # reply = response.choices[0].message.content

        for chunk in response:
            if chunk.choices[0].delta.content:
                streamed_text += chunk.choices[0].delta.content
                assistant_placeholder.markdown(streamed_text + "▌")

        assistant_placeholder.markdown(streamed_text)

        # Append assistant's final message to history
        st.session_state.messages.append(
            {"role": "assistant", "content": streamed_text}
        )

    except Exception as e:
        st.error(f"❌ Failed to connect to LLM: {e}")
