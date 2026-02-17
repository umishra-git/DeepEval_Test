import os
from deepeval.test_cases import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

# Load API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

def test_contextual_relevancy():
    """Test case using ContextualRelevancyMetric with LLMTestCase"""
    
    # Create an LLM Test Case
    test_case = LLMTestCase(
        input="What is the capital of Japan?",
        actual_output="Tokyo is the capital and largest city of Japan.",
        context="Japan is an island nation in East Asia. Tokyo is the capital and most populous city of Japan, located on the Kanto Plain."
    )
    
    # Initialize the ContextualRelevancyMetric
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)
    
    # Measure the metric
    contextual_relevancy_metric.measure(test_case)
    
    # Print results
    print(f"Metric: {contextual_relevancy_metric.metric_name}")
    print(f"Score: {contextual_relevancy_metric.score}")
    print(f"Reason: {contextual_relevancy_metric.reason}")
    
    # Assert relevancy meets threshold
    assert contextual_relevancy_metric.score >= 0.5, f"Contextual relevancy score {contextual_relevancy_metric.score} is below threshold"


def test_multiple_contextual_relevancy_cases():
    """Test multiple LLM test cases with ContextualRelevancyMetric"""
    
    test_cases = [
        LLMTestCase(
            input="What is photosynthesis?",
            actual_output="Photosynthesis is the process by which plants convert light energy into chemical energy.",
            context="Photosynthesis is a biochemical process that converts light energy into chemical energy stored in glucose. It occurs in plants, algae, and some bacteria."
        ),
        LLMTestCase(
            input="Who invented the telephone?",
            actual_output="Alexander Graham Bell is credited with inventing the telephone.",
            context="Alexander Graham Bell patented the telephone in 1876. The telephone was one of the most important inventions of the 19th century."
        ),
        LLMTestCase(
            input="What is the smallest planet in our solar system?",
            actual_output="Mercury is the smallest planet in our solar system.",
            context="Mercury is the smallest planet in our solar system and is closest to the Sun. It is named after the Roman messenger god."
        ),
    ]
    
    # Run evaluation on all test cases
    for test_case in test_cases:
        contextual_relevancy_metric = ContextualRelevancyMetric()
        contextual_relevancy_metric.measure(test_case)
        
        print(f"\nInput: {test_case.input}")
        print(f"Output: {test_case.actual_output}")
        print(f"Contextual Relevancy Score: {contextual_relevancy_metric.score}")
        
        assert contextual_relevancy_metric.score >= 0.5, f"Low contextual relevancy for input: {test_case.input}"


if __name__ == "__main__":
    test_contextual_relevancy()
    test_multiple_contextual_relevancy_cases()
    print("\nAll contextual relevancy tests passed!")
