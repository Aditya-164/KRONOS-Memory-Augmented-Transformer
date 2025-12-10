import re
from typing import Dict, Any, List, Optional

class LocalNeedleEvaluator:
    """A simple evaluator that doesn't require external API calls."""
    
    def __init__(self, model_name: str = "local-evaluator"):
        self.model_name = model_name
        print(f"Initialized local evaluator: {model_name}")
    
    def evaluate_retrieval(self, 
                          needle: str, 
                          retrieved_text: str, 
                          question: str,
                          response: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate if the needle was found and how relevant the response is.
        
        Args:
            needle: The text to find in the haystack
            retrieved_text: The context retrieved by the model
            question: The question asked about the needle
            response: The model's response to the question
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if needle is in retrieved text (exact or fuzzy match)
        contains_needle = self._check_needle_presence(needle, retrieved_text)
        
        # If we have a response, check its relevance to the question
        relevance_score = 0.0
        answer_quality = "unknown"
        
        if response:
            relevance_score = self._calculate_relevance(needle, question, response)
            answer_quality = self._evaluate_answer_quality(needle, question, response)
        
        # Return evaluation in expected format
        return {
            "contains_needle": contains_needle,
            "needle_found": "yes" if contains_needle else "no",
            "context_quality": self._score_context_quality(needle, retrieved_text),
            "answer_relevance": relevance_score,
            "answer_quality": answer_quality,
            "evaluator": "local-algorithm",
            "explanation": self._generate_explanation(needle, retrieved_text, question, response, contains_needle)
        }
    
    def _check_needle_presence(self, needle: str, text: str) -> bool:
        """Check if the needle is present in the text."""
        # Exact match check
        if needle.lower() in text.lower():
            return True
        
        # Fuzzy matching - check if most words from needle appear in text
        needle_words = set(re.findall(r'\b\w+\b', needle.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # If 80% of needle words are found, consider it a match
        common_words = needle_words.intersection(text_words)
        if len(needle_words) > 0 and len(common_words) / len(needle_words) >= 0.8:
            return True
            
        return False
    
    def _score_context_quality(self, needle: str, context: str) -> float:
        """Score how well the needle is represented in the context."""
        # Simple scoring: check what percentage of needle words appear
        needle_words = set(re.findall(r'\b\w+\b', needle.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        if not needle_words:
            return 0.5
            
        common_words = needle_words.intersection(context_words)
        return len(common_words) / len(needle_words)
    
    def _calculate_relevance(self, needle: str, question: str, response: str) -> float:
        """Calculate how relevant the response is to the question about the needle."""
        # Extract key terms from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words -= {"what", "when", "where", "who", "how", "why", "is", "are", "the", "a", "an"}
        
        # Extract key terms from needle
        needle_words = set(re.findall(r'\b\w+\b', needle.lower()))
        
        # Extract key terms from response
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Check overlap between question/needle terms and response
        relevant_terms = (question_words | needle_words)
        if not relevant_terms:
            return 0.5
            
        overlap = relevant_terms.intersection(response_words)
        
        # Calculate relevance score
        return min(1.0, len(overlap) / max(1, len(relevant_terms) * 0.5))
    
    def _evaluate_answer_quality(self, needle: str, question: str, response: str) -> str:
        """Evaluate the quality of the answer."""
        relevance = self._calculate_relevance(needle, question, response)
        
        if relevance > 0.8:
            return "excellent"
        elif relevance > 0.6:
            return "good"
        elif relevance > 0.4:
            return "fair"
        elif relevance > 0.2:
            return "poor"
        else:
            return "irrelevant"
    
    def _generate_explanation(self, needle: str, context: str, question: str, 
                             response: Optional[str], contains_needle: bool) -> str:
        """Generate an explanation for the evaluation."""
        if contains_needle:
            explanation = "The needle was found in the retrieved context. "
        else:
            explanation = "The needle was not found in the retrieved context. "
            
        if response:
            explanation += f"The response shows {self._evaluate_answer_quality(needle, question, response)} " \
                          f"relevance to the question about the needle."
        
        return explanation