import sys
import types

# Create a minimal Streamlit stub to allow import of the app
st = types.ModuleType("streamlit")

class _DummyCTX:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

# No-op functions
def _noop(*args, **kwargs):
    return None

# Structures returning simple objects
st.set_page_config = _noop
st.title = _noop
st.header = _noop
st.subheader = _noop
st.caption = _noop
st.success = _noop
st.error = _noop
st.warning = _noop
st.info = _noop
st.divider = _noop
st.markdown = _noop
st.write = _noop
st.metric = _noop
def _identity_decorator(*args, **kwargs):
    def _wrap(func):
        return func
    return _wrap
st.cache_data = _identity_decorator

# Widgets
def _selectbox(label, options, index=0, **kwargs):
    try:
        if options:
            if 0 <= index < len(options):
                return options[index]
            return options[0]
        return None
    except Exception:
        return options[0] if options else None
st.selectbox = _selectbox
st.slider = lambda *args, **kwargs: kwargs.get("value", 0.0)
st.checkbox = lambda *args, **kwargs: kwargs.get("value", False)
st.text_area = lambda *args, **kwargs: ""
st.button = lambda *args, **kwargs: False

# Containers and layouts
st.sidebar = _DummyCTX()
st.columns = lambda n: tuple(_DummyCTX() for _ in range(n))
st.container = lambda *a, **k: _DummyCTX()
st.spinner = lambda *a, **k: _DummyCTX()
st.expander = lambda *a, **k: _DummyCTX()

# Session state
st.session_state = {}

sys.modules["streamlit"] = st

# Stub optional libs used in app's model listing
ollama_stub = types.ModuleType("ollama")
ollama_stub.list = lambda: {"models": [{"name": "llama3:latest"}]}
sys.modules["ollama"] = ollama_stub

hf_stub = types.ModuleType("huggingface_hub")
class _FakeApi:
    def list_models(self, search: str, limit: int = 20):
        class _Obj:
            def __init__(self, modelId):
                self.modelId = modelId
        return [_Obj("google/flan-t5-small"), _Obj("google/flan-t5-base")]
hf_stub.HfApi = _FakeApi
sys.modules["huggingface_hub"] = hf_stub

# Now import the app
import importlib
# Stub internal src modules to avoid heavy imports
fake_evaluator = types.ModuleType("src.evaluator")
def _fake_calculate_metrics(reference, candidate):
    return {"bleu": 0.5, "rouge_l": 0.6, "bert_similarity": 0.9}
fake_evaluator.calculate_metrics = _fake_calculate_metrics
fake_evaluator.global_scorer = None
sys.modules["src.evaluator"] = fake_evaluator

fake_generator = types.ModuleType("src.generator")
def _fake_generate_documentation(**kwargs):
    return "Adds two numbers and returns the sum."
fake_generator.generate_documentation = _fake_generate_documentation
sys.modules["src.generator"] = fake_generator

fake_analysis = types.ModuleType("src.analysis")
def _fake_detect_hallucination(code, doc):
    return {"has_hallucination": False, "error_type": "No Error", "root_cause": "N/A"}
fake_analysis.detect_hallucination = _fake_detect_hallucination
sys.modules["src.analysis"] = fake_analysis

fake_bias = types.ModuleType("src.bias_analyzer")
class _FakeAnalyzer:
    def generate_accessibility_report(self, text, language="python"):
        class _R:
            flesch_reading_ease = 60.0
            flesch_kincaid_grade = 8.0
            gunning_fog = 10.0
        rep = fake_bias.AccessibilityReport()
        rep.overall_accessibility_score = 75.0
        rep.grade_level_interpretation = "Middle school level"
        rep.readability = _R()
        rep.vocabulary = types.SimpleNamespace()
        rep.recommendations = []
        return rep
fake_bias.BiasAnalyzer = _FakeAnalyzer
fake_bias.AccessibilityReport = type("AccessibilityReport", (), {})
sys.modules["src.bias_analyzer"] = fake_bias

app_module = importlib.import_module("src.app")


def test_app_imports_without_errors():
    # If we reached here, import was successful. Check helper functions exist.
    assert hasattr(app_module, "list_ollama_models")
    assert hasattr(app_module, "list_hf_models_curated")
