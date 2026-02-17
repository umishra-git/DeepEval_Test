import sys
import os

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

# Import all test modules
from test_relevantanswer import test_answer_relevancy, test_multiple_cases
from test_hallucination import test_hallucination_detection, test_multiple_hallucination_cases
from test_toxicity import test_toxicity_detection, test_multiple_toxicity_cases
from test_contextrelevancy import test_contextual_relevancy, test_multiple_contextual_relevancy_cases
from test_bias import test_bias_detection, test_multiple_bias_cases
from test_faithfulness import test_faithfulness, test_multiple_faithfulness_cases

def run_all_tests():
    """Run all tests from all test files"""
    
    tests = [
        ("Answer Relevancy - Single", test_answer_relevancy),
        ("Answer Relevancy - Multiple", test_multiple_cases),
        ("Hallucination - Single", test_hallucination_detection),
        ("Hallucination - Multiple", test_multiple_hallucination_cases),
        ("Toxicity - Single", test_toxicity_detection),
        ("Toxicity - Multiple", test_multiple_toxicity_cases),
        ("Contextual Relevancy - Single", test_contextual_relevancy),
        ("Contextual Relevancy - Multiple", test_multiple_contextual_relevancy_cases),
        ("Bias - Single", test_bias_detection),
        ("Bias - Multiple", test_multiple_bias_cases),
        ("Faithfulness - Single", test_faithfulness),
        ("Faithfulness - Multiple", test_multiple_faithfulness_cases),
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 80)
    print("RUNNING ALL DEEPEVAL TESTS")
    print("=" * 80)
    print()
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {str(e)}")
            failed += 1
        print()
    
    print("=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
