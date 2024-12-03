from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class LogprobComparison:
    """Enhanced comparison class that includes confidence metrics."""
    value: float
    logprob: float
    top_logprobs: List[Dict[str, float]]  # List of alternative values and their logprobs
    node_from: str
    node_a: str
    node_b: str
    context: str

    def confidence_score(self) -> float:
        """Calculate confidence score from logprobs."""
        # Convert logprob to linear probability
        primary_prob = np.exp(self.logprob)
        # Get second best probability if available
        second_best = max((np.exp(lp["logprob"]) for lp in self.top_logprobs[1:]), default=0)
        # Confidence is the difference between best and second best
        return primary_prob - second_best

    def weighted_value(self) -> float:
        """Get comparison value weighted by confidence."""
        conf = self.confidence_score()
        # If very confident (>0.9), use exact value
        if conf > 0.9:
            return self.value
        # Otherwise do weighted average with top alternatives
        weighted_sum = self.value * conf
        remaining_weight = 1.0 - conf
        for alt in self.top_logprobs[1:]:
            alt_prob = np.exp(alt["logprob"])
            weighted_sum += float(alt["value"]) * (alt_prob * remaining_weight)
        return weighted_sum

    @property
    def is_highly_confident(self) -> bool:
        """Check if this is a high confidence comparison."""
        return self.confidence_score() > 0.9