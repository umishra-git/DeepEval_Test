import os
from deepeval.test_cases import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

# Load API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

def test_faithfulness():
    """Test case using FaithfulnessMetric with LLMTestCase"""
    
    # Create an LLM Test Case
    test_case = LLMTestCase(
        input="What is the capital of Germany?",
        actual_output="Berlin is the capital of Germany.",
        context="Germany is a country in Central Europe. Berlin is the capital and largest city of Germany, located in northeastern Germany."
    )
    
    # Initialize the FaithfulnessMetric
    faithfulness_metric = FaithfulnessMetric(threshold=0.3)
    
    # Measure the metric
    faithfulness_metric.measure(test_case)
    
    # Print results
    print(f"Metric: {faithfulness_metric.metric_name}")
    print(f"Score: {faithfulness_metric.score}")
    print(f"Reason: {faithfulness_metric.reason}")
    
    # Assert faithfulness meets threshold
    assert faithfulness_metric.score >= 0.3, f"Faithfulness score {faithfulness_metric.score} is below threshold"


def test_multiple_faithfulness_cases():
    """Test multiple LLM test cases with FaithfulnessMetric"""
    
    test_cases = [
        LLMTestCase(
            input="How many continents are there?",
            actual_output="There are seven continents: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America.",
            context="The Earth is divided into seven continents. These are Africa, Antarctica, Asia, Europe, North America, Oceania (Australia), and South America."
        ),
        LLMTestCase(
            input="What is the largest mammal?",
            actual_output="The blue whale is the largest mammal on Earth.",
            context="The blue whale is the largest animal on Earth and the largest mammal ever known to have lived. It can grow up to 100 feet long."
        ),
        LLMTestCase(
            input="Who was the first president of the United States?",
            actual_output="George Washington was the first president of the United States, serving from 1789 to 1797.",
            context="George Washington was the first president of the United States. He served as president from 1789 until his retirement in 1797."
        ),
    ]
    
    # Run evaluation on all test cases
    for test_case in test_cases:
        faithfulness_metric = FaithfulnessMetric()
        faithfulness_metric.measure(test_case)
        
        print(f"\nInput: {test_case.input}")
        print(f"Output: {test_case.actual_output}")
        print(f"Faithfulness Score: {faithfulness_metric.score}")
        
        assert faithfulness_metric.score >= 0.5, f"Low faithfulness for input: {test_case.input}"


if __name__ == "__main__":
    test_faithfulness()
    test_multiple_faithfulness_cases()
    print("\nAll faithfulness tests passed!")
