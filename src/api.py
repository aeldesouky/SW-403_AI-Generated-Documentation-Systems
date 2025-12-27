from typing import Optional, Dict, Any, List
import time
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Prefer package imports; fall back if running directly
try:
    from src.generator import generate_documentation
except Exception:  # pragma: no cover
    from generator import generate_documentation  # type: ignore

# Safety utilities only for output filtering
try:
    from src.safety import filter_output, audit_log
except Exception:  # pragma: no cover
    from safety import filter_output, audit_log  # type: ignore

app = FastAPI(title="AI Documentation Service", version="1.0.0")

# CORS (optional, broad for ease of integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    code_snippet: str = Field(..., description="Source code to document")
    language: str = Field("Python", description="Language of the code")
    model_name: Optional[str] = Field("gpt-3.5-turbo", description="Model identifier")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Sampling temperature")
    provider: Optional[str] = Field(
        None,
        description="Provider: Bytez, OpenAI, Ollama, Hugging Face"
    )
    # Optional post-processing / evaluation
    ground_truth: Optional[str] = Field(
        None,
        description="Reference docstring to compute metrics"
    )
    run_hallucination_check: bool = Field(False, description="Run hallucination detection")
    run_accessibility_analysis: bool = Field(False, description="Run accessibility/readability analysis")


class ReadabilityMetrics(BaseModel):
    flesch_reading_ease: Optional[float] = None
    flesch_kincaid_grade: Optional[float] = None
    gunning_fog: Optional[float] = None


class AccessibilityReport(BaseModel):
    overall_accessibility_score: Optional[float] = None
    grade_level_interpretation: Optional[str] = None
    readability: Optional[ReadabilityMetrics] = None
    recommendations: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    docstring: str
    provider: Optional[str] = None
    model: Optional[str] = None
    language: str
    duration_ms: int
    metrics: Optional[Dict[str, float]] = None
    hallucination: Optional[Dict[str, Any]] = None
    accessibility: Optional[AccessibilityReport] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    start = time.time()

    # 1) Generate docstring via core generator
    try:
        doc = generate_documentation(
            code_snippet=req.code_snippet,
            language=req.language,
            model_name=req.model_name or "gpt-3.5-turbo",
            temperature=req.temperature,
            provider=req.provider,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Safety filter
    doc = filter_output(doc or "")

    out: Dict[str, Any] = {
        "docstring": doc,
        "provider": req.provider,
        "model": req.model_name,
        "language": req.language,
        "duration_ms": int((time.time() - start) * 1000),
    }

    # 2) Optional metrics
    if req.ground_truth:
        try:
            # Import lazily to avoid heavy init unless requested
            try:
                from src.evaluator import calculate_metrics
            except Exception:
                from evaluator import calculate_metrics  # type: ignore
            out["metrics"] = calculate_metrics(req.ground_truth, doc)
        except Exception as e:
            # Non-fatal; attach error marker
            out["metrics"] = {"error": f"metrics_failed: {e}"}

    # 3) Optional hallucination detection
    if req.run_hallucination_check:
        try:
            try:
                from src.analysis import detect_hallucination
            except Exception:
                from analysis import detect_hallucination  # type: ignore
            out["hallucination"] = detect_hallucination(req.code_snippet, doc)
        except Exception as e:
            out["hallucination"] = {"error": f"hallucination_failed: {e}"}

    # 4) Optional accessibility analysis
    if req.run_accessibility_analysis:
        try:
            try:
                from src.bias_analyzer import BiasAnalyzer
            except Exception:
                from bias_analyzer import BiasAnalyzer  # type: ignore
            analyzer = BiasAnalyzer()
            lang = (req.language or "python").lower()
            report = analyzer.generate_accessibility_report(doc, lang)
            out["accessibility"] = AccessibilityReport(
                overall_accessibility_score=getattr(report, "overall_accessibility_score", None),
                grade_level_interpretation=getattr(report, "grade_level_interpretation", None),
                readability=ReadabilityMetrics(
                    flesch_reading_ease=getattr(report.readability, "flesch_reading_ease", None),
                    flesch_kincaid_grade=getattr(report.readability, "flesch_kincaid_grade", None),
                    gunning_fog=getattr(report.readability, "gunning_fog", None),
                ) if getattr(report, "readability", None) else None,
                recommendations=(getattr(report, "recommendations", None) or [])[:5],
            )
        except Exception as e:
            out["accessibility"] = {"error": f"accessibility_failed: {e}"}

    # Audit (best-effort)
    try:
        audit_log("api_generate", {
            "provider": req.provider,
            "model": req.model_name,
            "language": req.language,
            "input_len": len(req.code_snippet or ""),
            "output_len": len(doc or ""),
        })
    except Exception:
        pass

    return GenerateResponse(**out)


# Convenience: root endpoint
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "AI Documentation Service. POST /generate"}
