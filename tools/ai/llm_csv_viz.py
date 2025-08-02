import streamlit as st
import pandas as pd
import google.generativeai as genai
import traceback
import io
import os

st.set_page_config(page_title="LLM CSV Visualizer", layout="wide")
st.title("📊 LLM-Powered CSV Visualizer with Gemini")

# API Key Input
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.sidebar.header("🔐 Gemini API Key")
api_key_input = st.sidebar.text_input("Enter your Gemini API Key", type="password")
if api_key_input:
    st.session_state.api_key = api_key_input
    genai.configure(api_key=api_key_input)

# CSV Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
df = None
filename = None

if uploaded_file:
    filename = uploaded_file.name
    try:
        df = pd.read_csv(uploaded_file)
        df.to_csv(f"/tmp/{filename}", index=False)
        st.success(f"✅ Loaded file: `{filename}`")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")

# Prompt Section
if df is not None and st.session_state.api_key:
    st.subheader("🧠 Ask Gemini to create a visualization")
    user_prompt = st.text_input(
        "Enter your prompt (e.g., 'Show a bar chart of total sales by category')"
    )

    execute_generated_code = st.checkbox("🔨 Execute the generated code", value=False)

    if user_prompt:
        table_description = (
            f"Filename: /tmp/{filename}\n"
            f"Columns: {list(df.columns)}\n\n"
            f"Sample Data:\n{df.head(5).to_csv(index=False)}"
        )

        full_prompt = (
            f"You are a helpful assistant that writes Python data visualization code "
            f"using plotly.express or matplotlib based on the user's prompt and the CSV file provided.\n\n"
            f"User prompt: {user_prompt}\n\n"
            f"CSV Info:\n{table_description}\n\n"
            f"Only respond with Python code in a code block (```). Don't explain it."
        )

        try:
            st.info("📡 Sending request to Gemini...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(full_prompt)
            content = response.text

            # Show raw markdown response
            st.subheader("📜 Gemini Response:")
            st.markdown(content)

            # Extract Python code
            code_block = ""
            in_code = False
            for line in content.splitlines():
                if "```" in line:
                    in_code = not in_code
                    continue
                if in_code:
                    code_block += line + "\n"

            if not code_block.strip():
                st.error("❌ No valid Python code was found.")
            else:
                st.subheader("🧪 Extracted Code")
                st.code(code_block, language="python")

                if execute_generated_code:
                    st.subheader("📈 Executed Chart")
                    try:
                        exec_globals = {
                            "df": df,
                            "pd": pd,
                            "st": st,
                            "__builtins__": __builtins__,
                            "px": __import__("plotly.express"),
                            "plt": __import__("matplotlib.pyplot"),
                        }
                        exec(code_block, exec_globals)
                    except Exception as exec_err:
                        st.error("⚠️ Error executing the generated code:")
                        st.exception(exec_err)
                else:
                    st.info(
                        "✅ Code execution skipped. Enable 'Execute the generated code' to run it."
                    )

        except Exception as e:
            st.error("🚨 Gemini API call failed:")
            st.exception(e)

elif df is not None:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
