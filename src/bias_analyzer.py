"""
Bias Analyzer Module
====================

This module analyzes AI-generated documentation for equity and accessibility,
addressing RQ5 of the research project: "Do AI documentation systems equitably
serve diverse user populations?"

Features:
- Readability metrics (Flesch-Kincaid, Gunning Fog, SMOG, ARI)
- Vocabulary complexity analysis
- Technical jargon detection
- Persona-based accessibility assessment
- Comprehensive accessibility reports with recommendations

Author: AI Documentation Systems Team
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

# Try to import textstat for advanced metrics
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Warning: textstat not installed. Using built-in readability calculations.")


class Persona(Enum):
    """User personas for accessibility analysis."""
    NOVICE_DEVELOPER = "novice_developer"
    EXPERT_DEVELOPER = "expert_developer"
    NON_NATIVE_SPEAKER = "non_native_speaker"
    TECHNICAL_WRITER = "technical_writer"


@dataclass
class ReadabilityScores:
    """Container for readability metric scores."""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    automated_readability_index: float
    coleman_liau_index: float
    avg_sentence_length: float
    avg_word_length: float
    
    def get_overall_grade_level(self) -> float:
        """Calculate average grade level across metrics."""
        return (
            self.flesch_kincaid_grade +
            self.gunning_fog +
            self.smog_index +
            self.automated_readability_index +
            self.coleman_liau_index
        ) / 5


@dataclass
class VocabularyAnalysis:
    """Container for vocabulary complexity analysis."""
    total_words: int
    unique_words: int
    vocabulary_richness: float  # Type-token ratio
    avg_syllables_per_word: float
    complex_word_ratio: float  # Words with 3+ syllables
    long_word_ratio: float  # Words with 7+ characters
    technical_term_count: int
    jargon_terms: List[str] = field(default_factory=list)


@dataclass
class PersonaAssessment:
    """Assessment results for a specific user persona."""
    persona: Persona
    accessibility_score: float  # 0-100
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AccessibilityReport:
    """Comprehensive accessibility report."""
    text_analyzed: str
    readability: ReadabilityScores
    vocabulary: VocabularyAnalysis
    persona_assessments: Dict[str, PersonaAssessment]
    overall_accessibility_score: float
    grade_level_interpretation: str
    recommendations: List[str]
    bias_indicators: List[str]


# Technical jargon dictionaries by language
JARGON_DICTIONARY = {
    "python": [
        "decorator", "generator", "comprehension", "iterator", "iterable",
        "metaclass", "mixin", "dunder", "magic method", "unpacking",
        "slicing", "lambda", "closure", "coroutine", "asyncio", "await",
        "yield", "namespace", "scope", "mutable", "immutable", "hashable",
        "args", "kwargs", "docstring", "pep", "pythonic", "stdlib",
        "virtualenv", "pip", "package", "module", "import", "exception",
        "traceback", "debugging", "profiling", "refactoring", "polymorphism",
        "inheritance", "encapsulation", "abstraction", "instantiation"
    ],
    "cobol": [
        "paragraph", "section", "division", "copybook", "perform", "thru",
        "compute", "working-storage", "linkage", "file-control", "fd",
        "pic", "picture", "redefines", "occurs", "indexed", "sequential",
        "vsam", "cics", "jcl", "batch", "mainframe", "ebcdic", "packed",
        "comp", "comp-3", "binary", "display", "group", "elementary",
        "filler", "level", "88-level", "condition-name", "copylib"
    ],
    "general": [
        "api", "sdk", "framework", "library", "dependency", "repository",
        "commit", "branch", "merge", "pull request", "ci/cd", "pipeline",
        "deployment", "containerization", "microservice", "monolith",
        "scalability", "latency", "throughput", "concurrency", "parallelism",
        "mutex", "semaphore", "deadlock", "race condition", "thread-safe",
        "idempotent", "stateless", "stateful", "singleton", "factory",
        "interface", "implementation", "abstraction", "polymorphism"
    ]
}


class BiasAnalyzer:
    """
    Analyzes documentation for accessibility, readability, and potential biases.
    
    This class provides comprehensive analysis to ensure AI-generated documentation
    serves diverse user populations equitably.
    """
    
    def __init__(self):
        """Initialize the BiasAnalyzer."""
        self.syllable_cache: Dict[str, int] = {}
    
    def count_syllables(self, word: str) -> int:
        """
        Count syllables in a word using a simple heuristic.
        
        Args:
            word: The word to count syllables for.
            
        Returns:
            Estimated number of syllables.
        """
        if word.lower() in self.syllable_cache:
            return self.syllable_cache[word.lower()]
        
        word = word.lower().strip()
        if len(word) <= 3:
            return 1
        
        # Remove trailing 'e' (silent e)
        if word.endswith('e'):
            word = word[:-1]
        
        # Remove trailing 'es' or 'ed' if preceded by consonant
        if word.endswith(('es', 'ed')) and len(word) > 4:
            word = word[:-2]
        
        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        count = max(1, count)
        self.syllable_cache[word.lower()] = count
        return count
    
    def tokenize(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Tokenize text into words and sentences.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            Tuple of (words list, sentences list)
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Split into words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        return words, sentences
    
    def analyze_readability(self, text: str) -> ReadabilityScores:
        """
        Calculate comprehensive readability metrics.
        
        Args:
            text: The documentation text to analyze.
            
        Returns:
            ReadabilityScores containing all metrics.
        """
        if TEXTSTAT_AVAILABLE:
            return self._analyze_readability_textstat(text)
        else:
            return self._analyze_readability_builtin(text)
    
    def _analyze_readability_textstat(self, text: str) -> ReadabilityScores:
        """Calculate readability using textstat library."""
        words, sentences = self.tokenize(text)
        
        return ReadabilityScores(
            flesch_reading_ease=textstat.flesch_reading_ease(text),
            flesch_kincaid_grade=textstat.flesch_kincaid_grade(text),
            gunning_fog=textstat.gunning_fog(text),
            smog_index=textstat.smog_index(text),
            automated_readability_index=textstat.automated_readability_index(text),
            coleman_liau_index=textstat.coleman_liau_index(text),
            avg_sentence_length=len(words) / max(len(sentences), 1),
            avg_word_length=sum(len(w) for w in words) / max(len(words), 1)
        )
    
    def _analyze_readability_builtin(self, text: str) -> ReadabilityScores:
        """Calculate readability using built-in formulas."""
        words, sentences = self.tokenize(text)
        
        if not words or not sentences:
            return ReadabilityScores(0, 0, 0, 0, 0, 0, 0, 0)
        
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(self.count_syllables(w) for w in words)
        total_characters = sum(len(w) for w in words)
        
        # Complex words (3+ syllables)
        complex_words = sum(1 for w in words if self.count_syllables(w) >= 3)
        
        # Flesch Reading Ease
        flesch_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        
        # Gunning Fog Index
        fog = 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))
        
        # SMOG Index
        smog = 1.0430 * math.sqrt(complex_words * (30 / total_sentences)) + 3.1291 if total_sentences >= 3 else 0
        
        # Automated Readability Index
        ari = 4.71 * (total_characters / total_words) + 0.5 * (total_words / total_sentences) - 21.43
        
        # Coleman-Liau Index
        L = (total_characters / total_words) * 100
        S = (total_sentences / total_words) * 100
        coleman = 0.0588 * L - 0.296 * S - 15.8
        
        return ReadabilityScores(
            flesch_reading_ease=max(0, min(100, flesch_ease)),
            flesch_kincaid_grade=max(0, fk_grade),
            gunning_fog=max(0, fog),
            smog_index=max(0, smog),
            automated_readability_index=max(0, ari),
            coleman_liau_index=max(0, coleman),
            avg_sentence_length=total_words / total_sentences,
            avg_word_length=total_characters / total_words
        )
    
    def analyze_vocabulary_complexity(self, text: str) -> VocabularyAnalysis:
        """
        Analyze vocabulary complexity of the documentation.
        
        Args:
            text: The documentation text to analyze.
            
        Returns:
            VocabularyAnalysis with complexity metrics.
        """
        words, _ = self.tokenize(text)
        
        if not words:
            return VocabularyAnalysis(0, 0, 0, 0, 0, 0, 0, [])
        
        total_words = len(words)
        unique_words = len(set(words))
        
        # Calculate metrics
        syllable_counts = [self.count_syllables(w) for w in words]
        avg_syllables = sum(syllable_counts) / total_words
        
        complex_words = sum(1 for s in syllable_counts if s >= 3)
        long_words = sum(1 for w in words if len(w) >= 7)
        
        return VocabularyAnalysis(
            total_words=total_words,
            unique_words=unique_words,
            vocabulary_richness=unique_words / total_words,
            avg_syllables_per_word=avg_syllables,
            complex_word_ratio=complex_words / total_words,
            long_word_ratio=long_words / total_words,
            technical_term_count=0,  # Updated by detect_jargon
            jargon_terms=[]
        )
    
    def detect_jargon(self, text: str, language: str = "python") -> Tuple[List[str], int]:
        """
        Detect technical jargon in the documentation.
        
        Args:
            text: The documentation text to analyze.
            language: Programming language context (python, cobol, general).
            
        Returns:
            Tuple of (list of jargon terms found, count)
        """
        text_lower = text.lower()
        found_jargon = []
        
        # Check language-specific jargon
        lang_jargon = JARGON_DICTIONARY.get(language.lower(), [])
        general_jargon = JARGON_DICTIONARY.get("general", [])
        
        all_jargon = set(lang_jargon + general_jargon)
        
        for term in all_jargon:
            # Check for whole word matches
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                found_jargon.append(term)
        
        return found_jargon, len(found_jargon)
    
    def analyze_for_persona(self, text: str, persona: Persona,
                           readability: Optional[ReadabilityScores] = None,
                           vocabulary: Optional[VocabularyAnalysis] = None) -> PersonaAssessment:
        """
        Assess documentation accessibility for a specific user persona.
        
        Args:
            text: The documentation text to analyze.
            persona: The user persona to assess for.
            readability: Pre-computed readability scores (optional).
            vocabulary: Pre-computed vocabulary analysis (optional).
            
        Returns:
            PersonaAssessment with score, issues, and recommendations.
        """
        if readability is None:
            readability = self.analyze_readability(text)
        if vocabulary is None:
            vocabulary = self.analyze_vocabulary_complexity(text)
        
        issues = []
        recommendations = []
        score = 100.0
        
        if persona == Persona.NOVICE_DEVELOPER:
            # Novices need simpler language, more explanations
            if readability.flesch_kincaid_grade > 10:
                score -= 20
                issues.append(f"Grade level too high ({readability.flesch_kincaid_grade:.1f}) for novice developers")
                recommendations.append("Simplify sentence structure and use more common words")
            
            if vocabulary.complex_word_ratio > 0.15:
                score -= 15
                issues.append(f"Too many complex words ({vocabulary.complex_word_ratio:.1%})")
                recommendations.append("Replace complex terminology with simpler alternatives")
            
            if vocabulary.technical_term_count > 5:
                score -= 10
                issues.append(f"High jargon density ({vocabulary.technical_term_count} terms)")
                recommendations.append("Define technical terms on first use or provide a glossary")
            
            if readability.avg_sentence_length > 20:
                score -= 10
                issues.append(f"Long sentences may confuse beginners (avg: {readability.avg_sentence_length:.1f} words)")
                recommendations.append("Break long sentences into shorter ones")
        
        elif persona == Persona.NON_NATIVE_SPEAKER:
            # Non-native speakers need clear, simple language
            if readability.flesch_reading_ease < 50:
                score -= 25
                issues.append(f"Low readability ({readability.flesch_reading_ease:.1f}) for non-native speakers")
                recommendations.append("Use simpler vocabulary and shorter sentences")
            
            if vocabulary.long_word_ratio > 0.20:
                score -= 15
                issues.append(f"Too many long words ({vocabulary.long_word_ratio:.1%})")
                recommendations.append("Replace long words with shorter synonyms where possible")
            
            # Check for idioms and colloquialisms
            idiom_patterns = [
                r'\bout of the box\b', r'\bunder the hood\b', r'\bboilerplate\b',
                r'\bfootgun\b', r'\bbikeshedding\b', r'\byak shaving\b'
            ]
            idioms_found = sum(1 for p in idiom_patterns if re.search(p, text.lower()))
            if idioms_found > 0:
                score -= 10 * idioms_found
                issues.append(f"Contains {idioms_found} idiom(s) that may confuse non-native speakers")
                recommendations.append("Replace idiomatic expressions with literal descriptions")
        
        elif persona == Persona.EXPERT_DEVELOPER:
            # Experts want concise, precise documentation
            if readability.flesch_reading_ease > 80 and len(text) > 100:
                score -= 10
                issues.append("Documentation may be overly simplistic for experts")
                recommendations.append("Include more technical details and edge cases")
            
            words, sentences = self.tokenize(text)
            if len(words) > 200:
                score -= 10
                issues.append("Documentation may be too verbose for experienced developers")
                recommendations.append("Consider condensing to essential information")
        
        elif persona == Persona.TECHNICAL_WRITER:
            # Technical writers look for consistency and completeness
            if readability.avg_sentence_length < 10 or readability.avg_sentence_length > 25:
                score -= 15
                issues.append(f"Inconsistent sentence length (avg: {readability.avg_sentence_length:.1f})")
                recommendations.append("Aim for 15-20 words per sentence for optimal clarity")
            
            if vocabulary.vocabulary_richness < 0.4:
                score -= 10
                issues.append("Low vocabulary variety - may indicate repetitive content")
                recommendations.append("Vary word choice to improve engagement")
        
        return PersonaAssessment(
            persona=persona,
            accessibility_score=max(0, min(100, score)),
            issues=issues,
            recommendations=recommendations
        )
    
    def generate_accessibility_report(self, text: str, 
                                      language: str = "python") -> AccessibilityReport:
        """
        Generate a comprehensive accessibility report for documentation.
        
        Args:
            text: The documentation text to analyze.
            language: Programming language context.
            
        Returns:
            Complete AccessibilityReport with all analyses and recommendations.
        """
        # Core analyses
        readability = self.analyze_readability(text)
        vocabulary = self.analyze_vocabulary_complexity(text)
        
        # Detect jargon
        jargon_terms, jargon_count = self.detect_jargon(text, language)
        vocabulary.jargon_terms = jargon_terms
        vocabulary.technical_term_count = jargon_count
        
        # Persona assessments
        persona_assessments = {}
        for persona in Persona:
            assessment = self.analyze_for_persona(text, persona, readability, vocabulary)
            persona_assessments[persona.value] = assessment
        
        # Calculate overall accessibility score
        overall_score = sum(a.accessibility_score for a in persona_assessments.values()) / len(persona_assessments)
        
        # Interpret grade level
        avg_grade = readability.get_overall_grade_level()
        if avg_grade < 6:
            grade_interpretation = "Elementary level - Very easy to understand"
        elif avg_grade < 9:
            grade_interpretation = "Middle school level - Easy to understand"
        elif avg_grade < 12:
            grade_interpretation = "High school level - Moderate difficulty"
        elif avg_grade < 16:
            grade_interpretation = "Undergraduate level - Challenging for non-experts"
        else:
            grade_interpretation = "Graduate level - Very difficult, experts only"
        
        # Compile recommendations
        all_recommendations = []
        for assessment in persona_assessments.values():
            all_recommendations.extend(assessment.recommendations)
        
        # Deduplicate recommendations
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        # Detect bias indicators
        bias_indicators = []
        
        # Check for gendered language
        gendered_terms = re.findall(r'\b(he|his|him|she|her|mankind|manpower)\b', text.lower())
        if gendered_terms:
            bias_indicators.append(f"Contains gendered language: {', '.join(set(gendered_terms))}")
        
        # Check for assumption of prior knowledge
        if jargon_count > 10:
            bias_indicators.append("High jargon density assumes significant prior knowledge")
        
        # Check for cultural references
        cultural_refs = re.findall(r'\b(western|american|english-speaking)\b', text.lower())
        if cultural_refs:
            bias_indicators.append(f"Contains cultural references: {', '.join(set(cultural_refs))}")
        
        return AccessibilityReport(
            text_analyzed=text[:500] + "..." if len(text) > 500 else text,
            readability=readability,
            vocabulary=vocabulary,
            persona_assessments=persona_assessments,
            overall_accessibility_score=overall_score,
            grade_level_interpretation=grade_interpretation,
            recommendations=unique_recommendations[:10],  # Top 10 recommendations
            bias_indicators=bias_indicators
        )
    
    def print_report(self, report: AccessibilityReport):
        """Pretty-print an accessibility report."""
        print("\n" + "=" * 70)
        print("ACCESSIBILITY ANALYSIS REPORT")
        print("=" * 70)
        
        print("\n📊 READABILITY SCORES:")
        r = report.readability
        print(f"  • Flesch Reading Ease: {r.flesch_reading_ease:.1f} ", end="")
        if r.flesch_reading_ease >= 60:
            print("✓ (Easy)")
        elif r.flesch_reading_ease >= 30:
            print("⚠ (Moderate)")
        else:
            print("✗ (Difficult)")
        
        print(f"  • Flesch-Kincaid Grade: {r.flesch_kincaid_grade:.1f}")
        print(f"  • Gunning Fog Index: {r.gunning_fog:.1f}")
        print(f"  • SMOG Index: {r.smog_index:.1f}")
        print(f"  • Coleman-Liau Index: {r.coleman_liau_index:.1f}")
        print(f"  • Avg Sentence Length: {r.avg_sentence_length:.1f} words")
        print(f"\n  📖 Interpretation: {report.grade_level_interpretation}")
        
        print("\n📝 VOCABULARY ANALYSIS:")
        v = report.vocabulary
        print(f"  • Total Words: {v.total_words}")
        print(f"  • Unique Words: {v.unique_words} ({v.vocabulary_richness:.1%} richness)")
        print(f"  • Complex Words (3+ syllables): {v.complex_word_ratio:.1%}")
        print(f"  • Long Words (7+ chars): {v.long_word_ratio:.1%}")
        print(f"  • Technical Jargon Found: {v.technical_term_count}")
        if v.jargon_terms:
            print(f"    → {', '.join(v.jargon_terms[:10])}")
        
        print("\n👥 PERSONA ACCESSIBILITY SCORES:")
        for persona_name, assessment in report.persona_assessments.items():
            emoji = "✓" if assessment.accessibility_score >= 70 else ("⚠" if assessment.accessibility_score >= 50 else "✗")
            print(f"  {emoji} {persona_name.replace('_', ' ').title()}: {assessment.accessibility_score:.0f}/100")
            if assessment.issues:
                for issue in assessment.issues[:2]:
                    print(f"      - {issue}")
        
        print(f"\n🎯 OVERALL ACCESSIBILITY SCORE: {report.overall_accessibility_score:.0f}/100")
        
        if report.bias_indicators:
            print("\n⚠️ BIAS INDICATORS:")
            for indicator in report.bias_indicators:
                print(f"  • {indicator}")
        
        if report.recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 70)


def demo():
    """Demonstrate the bias analyzer capabilities."""
    print("=" * 70)
    print("BIAS ANALYZER DEMO")
    print("=" * 70)
    
    # Sample documentation to analyze
    sample_docs = [
        {
            "name": "Good Example",
            "text": """Calculates the average of a list of numbers.
            
This function takes a list of numeric values and returns their mean.
If the list is empty, it returns zero to avoid division errors.

Parameters:
    numbers: A list of integers or floating-point values.

Returns:
    The arithmetic mean of all values in the list."""
        },
        {
            "name": "Jargon-Heavy Example",
            "text": """Implements a memoized recursive closure utilizing dynamic programming 
paradigms to achieve O(log n) amortized time complexity through lazy evaluation 
of thunks with tail-call optimization. The metaclass decorator provides polymorphic 
dispatch via the visitor pattern, leveraging duck typing and EAFP principles."""
        },
        {
            "name": "Biased Example",
            "text": """This function is designed for the average American programmer who 
understands Western software conventions. He should find it straightforward 
to implement using his preferred IDE. The boilerplate code handles edge cases 
under the hood, so even junior developers can use it out of the box."""
        }
    ]
    
    analyzer = BiasAnalyzer()
    
    for doc in sample_docs:
        print(f"\n\n{'#' * 70}")
        print(f"# Analyzing: {doc['name']}")
        print(f"{'#' * 70}")
        print(f"\nText:\n{doc['text'][:200]}...")
        
        report = analyzer.generate_accessibility_report(doc['text'])
        analyzer.print_report(report)


if __name__ == "__main__":
    demo()
