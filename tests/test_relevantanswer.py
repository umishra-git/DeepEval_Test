import os
import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Load API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

def test_answer_relevancy():
    """Test case using AnswerRelevancy metric with LLMTestCase"""
    
    # Create an LLM Test Case
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France and is located in the north-central part of the country.",
        expected_output="Paris"
    )
    
    # Initialize the AnswerRelevancy metric
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
    
    # Measure the metric
    answer_relevancy.measure(test_case)
    
    # Print results
    print(f"Metric: {answer_relevancy.metric_name}")
    print(f"Score: {answer_relevancy.score}")
    print(f"Reason: {answer_relevancy.reason}")
    
    # Assert the score meets threshold
    assert answer_relevancy.score >= 0.7, f"Answer relevancy score {answer_relevancy.score} is below threshold"


def test_multiple_cases():
    """Test multiple LLM test cases with AnswerRelevancy"""
    
    test_cases = [
        LLMTestCase(
            input="What is 2 + 2?",
            actual_output="The sum of 2 and 2 equals 4.",
            expected_output="4"
        ),
        LLMTestCase(
            input="Who wrote Romeo and Juliet?",
            actual_output="William Shakespeare is the renowned author of the tragedy Romeo and Juliet.",
            expected_output="William Shakespeare"
        ),
        LLMTestCase(
            input="What is the largest planet in our solar system?",
            actual_output="Jupiter is the largest planet in our solar system.",
            expected_output="Jupiter"
        ),
    ]
    
    # Run evaluation on all test cases
    for test_case in test_cases:
        answer_relevancy = AnswerRelevancyMetric()
        answer_relevancy.measure(test_case)
        
        print(f"\nInput: {test_case.input}")
        print(f"Output: {test_case.actual_output}")
        print(f"Score: {answer_relevancy.score}")
        
        assert answer_relevancy.score >= 0.7, f"Test failed for input: {test_case.input}"


if __name__ == "__main__":
    test_answer_relevancy()
    test_multiple_cases()
    print("\nAll tests passed!")
