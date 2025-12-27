import streamlit as st
import pandas as pd
from generator import generate_documentation
from evaluator import calculate_metrics
from analysis import detect_hallucination
from bias_analyzer import BiasAnalyzer
import time
import os

st.set_page_config(page_title="AI Code Documenter", layout="wide")

st.title("AI-Powered Legacy & Modern Code Documenter")
st.markdown("### SW 403 Project - Phase 2 Prototype")

BYTEZ_ACTIVE = os.getenv("BYTEZ_KEY") is not None and os.getenv("BYTEZ_KEY").strip() != ""

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    st.warning("No OpenAI API Key found. Please add it to your .env file.")

# Sidebar: Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Status Indicator
    if BYTEZ_ACTIVE:
        st.success("Connected via Bytez")
    elif os.getenv("OPENAI_API_KEY"):
        st.success("Connected via OpenAI")
    else:
        st.error("No API Key Found")
    model_choice = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    language = st.selectbox("Language", ["Python", "COBOL", "Java", "MUMPS"])
    
    st.divider()
    st.header("Analysis Options")
    run_hallucination_check = st.checkbox("Run Hallucination Detection", value=True)
    run_bias_analysis = st.checkbox("Run Accessibility Analysis", value=True)

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
                st.session_state['code'] = code_input
                st.session_state['language'] = language.lower()

# Results Column
with col2:
    st.subheader("AI Generated Documentation")
    if 'result' in st.session_state:
        st.code(st.session_state['result'], language="markdown")
        st.info(f"Generation Time: {st.session_state['time']:.2f}s")
        
        # 3. Calculate Metrics (If ground truth exists)
        if st.session_state['truth']:
            st.divider()
            st.subheader("Quality Metrics")
            scores = calculate_metrics(st.session_state['truth'], st.session_state['result'])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("BLEU (Precision)", f"{scores['bleu']:.4f}")
            m2.metric("ROUGE-L (Recall)", f"{scores['rouge_l']:.4f}")
            m3.metric("Semantic Sim (BERT)", f"{scores['bert_similarity']:.4f}")
            
            if scores['bert_similarity'] < 0.7:
                st.warning("Low semantic similarity detected. Check for hallucinations.")

# Analysis Section
if 'result' in st.session_state:
    st.divider()
    
    col3, col4 = st.columns(2)
    
    # Hallucination Detection
    with col3:
        if run_hallucination_check:
            st.subheader("Hallucination Detection")
            with st.spinner("Analyzing for hallucinations..."):
                hall_result = detect_hallucination(
                    st.session_state.get('code', ''),
                    st.session_state['result']
                )
            
            if hall_result['has_hallucination']:
                st.error(f"Hallucination Detected: {hall_result['error_type']}")
                st.write(f"**Root Cause:** {hall_result['root_cause']}")
            else:
                st.success("No Hallucination Detected")
                st.write(f"**Status:** {hall_result['error_type']}")
    
    # Bias/Accessibility Analysis
    with col4:
        if run_bias_analysis:
            st.subheader("Accessibility Analysis")
            analyzer = BiasAnalyzer()
            lang = st.session_state.get('language', 'python')
            report = analyzer.generate_accessibility_report(st.session_state['result'], lang)
            
            # Score display
            score = report.overall_accessibility_score
            if score >= 80:
                st.success(f"Accessibility Score: {score:.0f}/100")
            elif score >= 60:
                st.warning(f"Accessibility Score: {score:.0f}/100")
            else:
                st.error(f"Accessibility Score: {score:.0f}/100")
            
            st.write(f"**Grade Level:** {report.grade_level_interpretation}")
            
            # Readability metrics
            with st.expander("Readability Details"):
                r = report.readability
                st.write(f"- Flesch Reading Ease: {r.flesch_reading_ease:.1f}")
                st.write(f"- Flesch-Kincaid Grade: {r.flesch_kincaid_grade:.1f}")
                st.write(f"- Gunning Fog Index: {r.gunning_fog:.1f}")
            
            # Recommendations
            if report.recommendations:
                with st.expander("Recommendations"):
                    for rec in report.recommendations[:5]:
                        st.write(f"- {rec}")
