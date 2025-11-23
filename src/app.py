import streamlit as st
import pandas as pd
from generator import generate_documentation
from evaluator import calculate_metrics
import time
import os

st.set_page_config(page_title="AI Code Documenter", layout="wide")

st.title("🤖 AI-Powered Legacy & Modern Code Documenter")
st.markdown("### SW 403 Project - Phase 2 Prototype")

BYTEZ_ACTIVE = os.getenv("BYTEZ_KEY") is not None and os.getenv("BYTEZ_KEY").strip() != ""

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ No OpenAI API Key found. Please add it to your .env file.")

# Sidebar: Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Status Indicator
    if BYTEZ_ACTIVE:
        st.success("🟢 Connected via Bytez")
    elif os.getenv("OPENAI_API_KEY"):
        st.success("🟢 Connected via OpenAI")
    else:
        st.error("🔴 No API Key Found")
    model_choice = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    language = st.selectbox("Language", ["Python", "COBOL", "Java", "MUMPS"])

# Main Area: Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Source Code")
    code_input = st.text_area("Paste code here...", height=300)
    
    st.subheader("Ground Truth (Optional)")
    st.caption("Paste the 'Correct' docstring here to calculate metrics.")
    ground_truth = st.text_area("Expected Docstring...", height=150)

    if st.button("Generate Documentation"):
        if not code_input:
            st.error("Please enter code first.")
        else:
            with st.spinner("Analyzing Logic..."):
                start_time = time.time()
                # 1. Generate
                generated_doc = generate_documentation(code_input, language, model_choice, temp)
                end_time = time.time()
                
                # 2. Display Result
                st.session_state['result'] = generated_doc
                st.session_state['time'] = end_time - start_time
                st.session_state['truth'] = ground_truth

# Results Column
with col2:
    st.subheader("AI Generated Documentation")
    if 'result' in st.session_state:
        st.code(st.session_state['result'], language="markdown")
        st.info(f"Generation Time: {st.session_state['time']:.2f}s")
        
        # 3. Calculate Metrics (If ground truth exists)
        if st.session_state['truth']:
            st.divider()
            st.subheader("🏆 Quality Metrics")
            scores = calculate_metrics(st.session_state['truth'], st.session_state['result'])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("BLEU (Precision)", f"{scores['bleu']:.4f}")
            m2.metric("ROUGE-L (Recall)", f"{scores['rouge_l']:.4f}")
            m3.metric("Semantic Sim (BERT)", f"{scores['bert_similarity']:.4f}")
            
            if scores['bert_similarity'] < 0.7:
                st.warning("⚠️ Low semantic similarity detected. Check for hallucinations.")
