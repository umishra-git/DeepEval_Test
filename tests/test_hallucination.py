import os
from deepeval.test_cases import LLMTestCase
from deepeval.metrics import HallucinationMetric

# Load API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

def test_hallucination_detection():
    """Test case using HallucinationMetric with LLMTestCase"""
    
    # Create an LLM Test Case
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        context="France is a country in Western Europe. Paris is the capital and largest city of France."
    )
    
    # Initialize the HallucinationMetric
    hallucination_metric = HallucinationMetric(threshold=0.5)
    
    # Measure the metric
    hallucination_metric.measure(test_case)
    
    # Print results
    print(f"Metric: {hallucination_metric.metric_name}")
    print(f"Score: {hallucination_metric.score}")
    print(f"Reason: {hallucination_metric.reason}")
    
    # Assert no hallucination detected
    assert hallucination_metric.score >= 0.5, f"Hallucination detected with score {hallucination_metric.score}"


def test_multiple_hallucination_cases():
    """Test multiple LLM test cases with HallucinationMetric"""
    
    test_cases = [
        LLMTestCase(
            input="Who is the current president of the United States?",
            actual_output="Joe Biden is the current president of the United States.",
            context="Joe Biden became the 46th president of the United States on January 20, 2021."
        ),
        LLMTestCase(
            input="What is the largest ocean?",
            actual_output="The Pacific Ocean is the largest ocean on Earth.",
            context="The Pacific Ocean is the largest ocean on Earth, covering an area of about 165 million square kilometers."
        ),
        LLMTestCase(
            input="When was the Eiffel Tower built?",
            actual_output="The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
            context="The Eiffel Tower was constructed from January 28, 1887 to March 31, 1889, for the 1889 World's Fair."
        ),
    ]
    
    # Run evaluation on all test cases
    for test_case in test_cases:
        hallucination_metric = HallucinationMetric(threshold=0.5)
        hallucination_metric.measure(test_case)
        
        print(f"\nInput: {test_case.input}")
        print(f"Output: {test_case.actual_output}")
        print(f"Hallucination Score: {hallucination_metric.score}")
        
        assert hallucination_metric.score >= 0.5, f"Hallucination detected for input: {test_case.input}"


if __name__ == "__main__":
    test_hallucination_detection()
    test_multiple_hallucination_cases()
    print("\nAll hallucination tests passed!")
