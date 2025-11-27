"""
Detecting Hallucinations & Security Vulnerabilities
====================================================

This module implements comprehensive testing for:
1. Hallucination Detection in AI-generated documentation
2. Security Vulnerability Scanning using Bandit and Semgrep
3. Code Quality Analysis
4. Comprehensive Reporting

Based on Lab 4 Presentation Requirements
Author: AI Documentation Systems Team
Date: November 27, 2025
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generator import generate_documentation
from evaluator import calculate_metrics
from analysis import detect_hallucination


class SecurityScanner:
    """Handles security vulnerability scanning using Bandit and Semgrep"""
    
    def __init__(self):
        self.results = {
            'bandit': {},
            'semgrep': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def check_tools_installed(self) -> Dict[str, bool]:
        """Check if security tools are installed"""
        tools_status = {}
        
        # Check Bandit
        try:
            result = subprocess.run(['bandit', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            tools_status['bandit'] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools_status['bandit'] = False
        
        # Check Semgrep
        try:
            result = subprocess.run(['semgrep', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            tools_status['semgrep'] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools_status['semgrep'] = False
        
        return tools_status
    
    def scan_with_bandit(self, code: str, language: str) -> Dict[str, Any]:
        """
        Scan code for security vulnerabilities using Bandit
        Only works for Python code
        """
        if language.lower() != 'python':
            return {
                'applicable': False,
                'reason': 'Bandit only supports Python code',
                'vulnerabilities': []
            }
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            # Run Bandit
            result = subprocess.run(
                ['bandit', '-f', 'json', tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(tmp_path)
            
            # Parse results
            if result.stdout:
                bandit_output = json.loads(result.stdout)
                vulnerabilities = bandit_output.get('results', [])
                
                return {
                    'applicable': True,
                    'total_issues': len(vulnerabilities),
                    'high_severity': sum(1 for v in vulnerabilities if v.get('issue_severity') == 'HIGH'),
                    'medium_severity': sum(1 for v in vulnerabilities if v.get('issue_severity') == 'MEDIUM'),
                    'low_severity': sum(1 for v in vulnerabilities if v.get('issue_severity') == 'LOW'),
                    'vulnerabilities': vulnerabilities[:5],  # Keep top 5 for report
                    'metrics': bandit_output.get('metrics', {}),
                    'status': 'success'
                }
            else:
                return {
                    'applicable': True,
                    'status': 'no_issues',
                    'total_issues': 0,
                    'vulnerabilities': []
                }
                
        except subprocess.TimeoutExpired:
            return {'applicable': True, 'status': 'timeout', 'error': 'Bandit scan timed out'}
        except Exception as e:
            return {'applicable': True, 'status': 'error', 'error': str(e)}
    
    def scan_with_semgrep(self, code: str, language: str) -> Dict[str, Any]:
        """
        Scan code for security vulnerabilities using Semgrep
        Supports multiple languages
        """
        try:
            # Map language names
            lang_map = {
                'python': 'py',
                'cobol': 'cobol',
                'java': 'java',
                'javascript': 'js',
                'typescript': 'ts'
            }
            
            file_ext = lang_map.get(language.lower(), 'txt')
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_ext}', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            # Run Semgrep with auto config
            result = subprocess.run(
                ['semgrep', '--config=auto', '--json', tmp_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Clean up
            os.unlink(tmp_path)
            
            # Parse results
            if result.stdout:
                semgrep_output = json.loads(result.stdout)
                findings = semgrep_output.get('results', [])
                
                return {
                    'applicable': True,
                    'total_findings': len(findings),
                    'error_count': sum(1 for f in findings if f.get('extra', {}).get('severity') == 'ERROR'),
                    'warning_count': sum(1 for f in findings if f.get('extra', {}).get('severity') == 'WARNING'),
                    'info_count': sum(1 for f in findings if f.get('extra', {}).get('severity') == 'INFO'),
                    'findings': findings[:5],  # Keep top 5 for report
                    'status': 'success'
                }
            else:
                return {
                    'applicable': True,
                    'status': 'no_findings',
                    'total_findings': 0,
                    'findings': []
                }
                
        except subprocess.TimeoutExpired:
            return {'applicable': True, 'status': 'timeout', 'error': 'Semgrep scan timed out'}
        except FileNotFoundError:
            return {'applicable': False, 'status': 'not_installed', 'error': 'Semgrep not found'}
        except Exception as e:
            return {'applicable': True, 'status': 'error', 'error': str(e)}


class HallucinationDetector:
    """Handles hallucination detection in generated documentation"""
    
    def __init__(self):
        self.results = []
    
    def detect_hallucinations(self, code: str, generated_doc: str, 
                            ground_truth: str = None) -> Dict[str, Any]:
        """
        Detect hallucinations using LLM-as-a-Judge approach
        """
        try:
            # Use the existing analysis module
            hallucination_result = detect_hallucination(code, generated_doc)
            
            result = {
                'has_hallucination': hallucination_result.get('has_hallucination', False),
                'error_type': hallucination_result.get('error_type', 'No Error'),
                'root_cause': hallucination_result.get('root_cause', 'No error'),
                'status': 'success'
            }
            
            # If ground truth is provided, calculate semantic similarity
            if ground_truth:
                metrics = calculate_metrics(ground_truth, generated_doc)
                result['semantic_metrics'] = {
                    'bert_similarity': metrics.get('bert_similarity', 0.0),
                    'bleu_score': metrics.get('bleu', 0.0),
                    'rouge_l': metrics.get('rouge_l', 0.0)
                }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'has_hallucination': False
            }


class ComprehensiveTester:
    """Main testing orchestrator"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.security_scanner = SecurityScanner()
        self.hallucination_detector = HallucinationDetector()
        
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'summary': {},
            'tool_status': {}
        }
    
    def check_environment(self) -> Dict[str, bool]:
        """Check if all required tools and APIs are available"""
        print("\n" + "="*70)
        print("ENVIRONMENT CHECK")
        print("="*70)
        
        # Check API keys
        from dotenv import load_dotenv
        load_dotenv()
        
        env_status = {
            'openai_api': os.getenv('OPENAI_API_KEY') is not None,
            'bytez_api': os.getenv('BYTEZ_KEY') is not None,
        }
        
        # Check security tools
        tool_status = self.security_scanner.check_tools_installed()
        env_status.update(tool_status)
        
        # Print status
        print("\nAPI Keys:")
        print(f"  ✓ OpenAI API: {'Available' if env_status['openai_api'] else '✗ Missing'}")
        print(f"  ✓ Bytez API: {'Available' if env_status['bytez_api'] else '✗ Missing'}")
        
        print("\nSecurity Tools:")
        print(f"  {'✓' if env_status.get('bandit') else '✗'} Bandit: {'Installed' if env_status.get('bandit') else 'Not Installed'}")
        print(f"  {'✓' if env_status.get('semgrep') else '✗'} Semgrep: {'Installed' if env_status.get('semgrep') else 'Not Installed'}")
        
        self.test_results['tool_status'] = env_status
        return env_status
    
    def run_single_test(self, test_data: Dict[str, Any], test_id: int) -> Dict[str, Any]:
        """Run comprehensive test on a single code sample"""
        
        code = test_data['code']
        language = test_data['language']
        ground_truth = test_data.get('ground_truth', '')
        
        print(f"\n{'='*70}")
        print(f"Test #{test_id}: {language} - {test_data.get('id', 'unknown')}")
        print(f"{'='*70}")
        
        result = {
            'test_id': test_id,
            'sample_id': test_data.get('id', 'unknown'),
            'language': language,
            'code_length': len(code),
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Generate Documentation
        print("\n[1/4] Generating documentation...")
        try:
            generated_doc = generate_documentation(code, language)
            result['generated_documentation'] = generated_doc
            result['generation_status'] = 'success'
            print(f"  ✓ Generated {len(generated_doc)} characters")
        except Exception as e:
            result['generation_status'] = 'failed'
            result['generation_error'] = str(e)
            print(f"  ✗ Generation failed: {e}")
            return result
        
        # Step 2: Detect Hallucinations
        print("\n[2/4] Detecting hallucinations...")
        try:
            hallucination_result = self.hallucination_detector.detect_hallucinations(
                code, generated_doc, ground_truth
            )
            result['hallucination_detection'] = hallucination_result
            
            if hallucination_result.get('has_hallucination'):
                print(f"  ⚠ HALLUCINATION DETECTED: {hallucination_result.get('error_type')}")
                print(f"    Reason: {hallucination_result.get('root_cause')}")
            else:
                print(f"  ✓ No hallucinations detected")
            
            # Print semantic metrics if available
            if 'semantic_metrics' in hallucination_result:
                metrics = hallucination_result['semantic_metrics']
                print(f"  📊 Semantic Similarity: {metrics['bert_similarity']:.3f}")
                print(f"     BLEU: {metrics['bleu_score']:.3f} | ROUGE-L: {metrics['rouge_l']:.3f}")
                
        except Exception as e:
            result['hallucination_detection'] = {'status': 'error', 'error': str(e)}
            print(f"  ✗ Hallucination detection failed: {e}")
        
        # Step 3: Security Scan with Bandit
        print("\n[3/4] Running Bandit security scan...")
        try:
            bandit_result = self.security_scanner.scan_with_bandit(code, language)
            result['bandit_scan'] = bandit_result
            
            if bandit_result.get('applicable'):
                if bandit_result.get('status') == 'success':
                    total = bandit_result.get('total_issues', 0)
                    high = bandit_result.get('high_severity', 0)
                    medium = bandit_result.get('medium_severity', 0)
                    low = bandit_result.get('low_severity', 0)
                    
                    if total > 0:
                        print(f"  ⚠ Found {total} issues: {high} HIGH, {medium} MEDIUM, {low} LOW")
                    else:
                        print(f"  ✓ No security issues found")
                else:
                    print(f"  ℹ {bandit_result.get('status')}")
            else:
                print(f"  ℹ {bandit_result.get('reason', 'Not applicable')}")
                
        except Exception as e:
            result['bandit_scan'] = {'status': 'error', 'error': str(e)}
            print(f"  ✗ Bandit scan failed: {e}")
        
        # Step 4: Security Scan with Semgrep
        print("\n[4/4] Running Semgrep security scan...")
        try:
            semgrep_result = self.security_scanner.scan_with_semgrep(code, language)
            result['semgrep_scan'] = semgrep_result
            
            if semgrep_result.get('applicable'):
                if semgrep_result.get('status') == 'success':
                    total = semgrep_result.get('total_findings', 0)
                    errors = semgrep_result.get('error_count', 0)
                    warnings = semgrep_result.get('warning_count', 0)
                    
                    if total > 0:
                        print(f"  ⚠ Found {total} findings: {errors} ERRORS, {warnings} WARNINGS")
                    else:
                        print(f"  ✓ No security findings")
                else:
                    print(f"  ℹ {semgrep_result.get('status')}")
            else:
                print(f"  ℹ {semgrep_result.get('reason', 'Not applicable')}")
                
        except Exception as e:
            result['semgrep_scan'] = {'status': 'error', 'error': str(e)}
            print(f"  ✗ Semgrep scan failed: {e}")
        
        return result
    
    def run_test_suite(self, dataset_path: str, max_samples: int = 10):
        """Run comprehensive tests on dataset"""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE TESTING SUITE")
        print("Detecting Hallucinations & Security Vulnerabilities")
        print("="*70)
        
        # Check environment
        self.check_environment()
        
        # Load dataset
        print(f"\n📂 Loading dataset from: {dataset_path}")
        test_samples = []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    test_samples.append(json.loads(line))
            
            print(f"✓ Loaded {len(test_samples)} test samples")
            
        except FileNotFoundError:
            print(f"✗ Dataset not found: {dataset_path}")
            return
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return
        
        # Run tests
        print(f"\n🚀 Running tests on {len(test_samples)} samples...")
        print("="*70)
        
        for i, sample in enumerate(test_samples, 1):
            try:
                test_result = self.run_single_test(sample, i)
                self.test_results['tests'].append(test_result)
            except Exception as e:
                print(f"\n✗ Test #{i} crashed: {e}")
                self.test_results['tests'].append({
                    'test_id': i,
                    'status': 'crashed',
                    'error': str(e)
                })
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        # Print final summary
        self.print_summary()
    
    def generate_summary(self):
        """Generate summary statistics"""
        tests = self.test_results['tests']
        
        summary = {
            'total_tests': len(tests),
            'successful_generations': sum(1 for t in tests if t.get('generation_status') == 'success'),
            'hallucinations_detected': sum(1 for t in tests 
                                          if t.get('hallucination_detection', {}).get('has_hallucination', False)),
            'total_security_issues': 0,
            'languages': {}
        }
        
        # Count by language
        for test in tests:
            lang = test.get('language', 'unknown')
            if lang not in summary['languages']:
                summary['languages'][lang] = {
                    'count': 0,
                    'hallucinations': 0,
                    'security_issues': 0
                }
            
            summary['languages'][lang]['count'] += 1
            
            if test.get('hallucination_detection', {}).get('has_hallucination', False):
                summary['languages'][lang]['hallucinations'] += 1
            
            # Count security issues
            bandit_issues = test.get('bandit_scan', {}).get('total_issues', 0)
            semgrep_issues = test.get('semgrep_scan', {}).get('total_findings', 0)
            total_issues = bandit_issues + semgrep_issues
            
            summary['total_security_issues'] += total_issues
            summary['languages'][lang]['security_issues'] += total_issues
        
        self.test_results['summary'] = summary
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"test_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def print_summary(self):
        """Print final summary"""
        summary = self.test_results['summary']
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total Tests: {summary['total_tests']}")
        print(f"  • Successful Generations: {summary['successful_generations']}")
        print(f"  • Hallucinations Detected: {summary['hallucinations_detected']}")
        print(f"  • Total Security Issues: {summary['total_security_issues']}")
        
        print(f"\n📈 By Language:")
        for lang, stats in summary['languages'].items():
            print(f"\n  {lang}:")
            print(f"    • Samples: {stats['count']}")
            print(f"    • Hallucinations: {stats['hallucinations']}")
            print(f"    • Security Issues: {stats['security_issues']}")
        
        print("\n" + "="*70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Testing for AI Documentation System'
    )
    parser.add_argument(
        '--dataset',
        default='data/processed/experiment_set.jsonl',
        help='Path to dataset file'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples to test'
    )
    parser.add_argument(
        '--output',
        default='test_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = ComprehensiveTester(output_dir=args.output)
    tester.run_test_suite(args.dataset, max_samples=args.samples)


if __name__ == "__main__":
    main()
