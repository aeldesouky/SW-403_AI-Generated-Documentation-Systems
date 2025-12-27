import streamlit as st
import pandas as pd
# Support both package and script execution contexts
try:
    from src.generator import generate_documentation
    from src.evaluator import calculate_metrics
    from src.analysis import detect_hallucination
    from src.bias_analyzer import BiasAnalyzer
except Exception:
    from generator import generate_documentation  # type: ignore
    from evaluator import calculate_metrics  # type: ignore
    from analysis import detect_hallucination  # type: ignore
    from bias_analyzer import BiasAnalyzer  # type: ignore
import time
import os
import json
from urllib.request import urlopen, Request
from urllib.error import URLError

# Optional integrations
try:
    import ollama  # For local model listing
except Exception:
    ollama = None

try:
    from huggingface_hub import HfApi  # For curated model discovery
except Exception:
    HfApi = None

st.set_page_config(page_title="AI Code Documenter", layout="wide")

st.title("AI Documentatiaon Generator and Analyzer")

BYTEZ_ACTIVE = os.getenv("BYTEZ_KEY") is not None and os.getenv("BYTEZ_KEY").strip() != ""
OPENAI_ACTIVE = os.getenv("OPENAI_API_KEY") is not None and os.getenv("OPENAI_API_KEY").strip() != ""


@st.cache_data(ttl=10, show_spinner=False)
def list_ollama_models():
    """Return available Ollama models (local server required)."""
    if ollama is None:
        # Fall back directly to REST if python package is not available
        pass
    else:
        try:
            tags = ollama.list()
            models = []
            if isinstance(tags, dict) and 'models' in tags:
                models = [m.get('name') for m in tags['models'] if m.get('name')]
            elif isinstance(tags, list):
                models = [m.get('name') for m in tags if isinstance(m, dict) and m.get('name')]
            if models:
                return sorted(set([m for m in models if isinstance(m, str)]))
        except Exception:
            # Will try REST fallback below
            pass

    # REST fallback to Ollama server
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = f"{host.rstrip('/')}/api/tags"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = []
        if isinstance(data, dict) and 'models' in data:
            models = [m.get('name') for m in data['models'] if m.get('name')]
        return sorted(set([m for m in models if isinstance(m, str)]))
    except URLError:
        return []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def list_hf_models_curated():
    """Curated Hugging Face models: Google FLAN + lightweight picks."""
    curated = [
        # Google FLAN-T5 variants (lightweight first)
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        # Other lightweight / small chat-capable models
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "google/gemma-2-2b-it",
        "distilgpt2",
    ]

    # Attempt to expand FLAN list via Hub search (optional)
    try:
        if HfApi is not None:
            api = HfApi()
            flan = api.list_models(search="google/flan-t5", limit=20)
            for m in flan:
                name = getattr(m, 'modelId', None)
                if name and name.startswith("google/flan-t5"):
                    curated.append(name)
    except Exception:
        pass

    # De-duplicate while preserving order
    seen = set()
    out = []
    for m in curated:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out

def get_provider_default_index():
    """Choose sensible default provider based on configured keys."""
    if BYTEZ_ACTIVE:
        return 0  # Bytez
    if OPENAI_ACTIVE:
        return 1  # OpenAI
    return 2  # Ollama as a local default

# Sidebar: Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Provider selection
    provider_choice = st.selectbox(
        "Provider",
        ["Bytez", "OpenAI", "Ollama", "Hugging Face"],
        index=get_provider_default_index()
    )

    # API/Provider status indicator
    if provider_choice == "Bytez":
        if BYTEZ_ACTIVE:
            st.success("Connected via Bytez")
        else:
            st.error("Bytez not configured (.env BYTEZ_KEY)")
    elif provider_choice == "OpenAI":
        if OPENAI_ACTIVE:
            st.success("Connected via OpenAI")
        else:
            st.error("OpenAI not configured (.env OPENAI_API_KEY)")
    elif provider_choice == "Ollama":
        ollama_models = list_ollama_models()
        if ollama_models:
            st.success(f"Ollama available: {len(ollama_models)} models")
        else:
            st.warning("Ollama not reachable or no models found")
    else:  # Hugging Face
        st.info("Using Hugging Face Hub for model list")

    # Dynamic model list and selection per provider
    if provider_choice in ("Bytez", "OpenAI"):
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        model_choice = st.selectbox("Select Model", model_options)
    elif provider_choice == "Ollama":
        ollama_models = list_ollama_models()
        if ollama_models:
            model_choice = st.selectbox("Select Model", ollama_models)
        else:
            st.error("No local Ollama models found. Install models with 'ollama pull <model>'.")
            model_choice = None
    else:  # Hugging Face
        model_options = list_hf_models_curated()
        model_choice = st.selectbox("Select Model", model_options)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    language = st.selectbox("Language", ["Python"])

    # Persist selection
    st.session_state['provider'] = provider_choice
    
    st.divider()
    st.header("Analysis Options")
    run_hallucination_check = st.checkbox("Run Hallucination Detection", value=True)
    run_bias_analysis = st.checkbox("Run Accessibility Analysis", value=True)

# Main Area: Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Source Code")
    code_input = st.text_area("Paste code here...", height=150)
    
    st.subheader("Ground Truth (Optional)")
    st.caption("Paste the 'Correct' docstring here to calculate metrics.")
    ground_truth = st.text_area("Expected Docstring...", height=150)

    if st.button("Generate Documentation"):
        if not code_input:
            st.error("Please enter code first.")
        else:
            with st.spinner("Analyzing Logic..."):
                start_time = time.time()
                # 1. Generate via selected provider
                provider_choice = st.session_state.get('provider')
                if provider_choice == "Ollama" and not model_choice:
                    st.error("Select an Ollama model before generating.")
                    generated_doc = ""
                else:
                    generated_doc = generate_documentation(
                        code_input,
                        language,
                        model_name=model_choice or "",
                        temperature=temp,
                        provider=provider_choice,
                    )
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
        # Display as wrapped text in a container
        with st.container(border=True):
            st.markdown(st.session_state['result'])
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