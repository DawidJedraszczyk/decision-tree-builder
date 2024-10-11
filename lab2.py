import math
from collections import Counter

class DecisionTreeInductor:
    def __init__(self, data: dict = None):
        """
        :param data:
        ex. {'low': ['no', 'no'], 'med': ['yes', 'no'], 'high': ['yes', 'yes', 'yes', 'no', 'yes', 'no']}
        """
        self.data = data or {
            'low': ['no', 'no'],
            'med': ['yes', 'no'],
            'high': ['yes', 'yes', 'yes', 'no', 'yes', 'no']
        }

        # Flatten decisions and initialize variables
        self.decisions = [decision for decision_list in self.data.values() for decision in decision_list]
        self.decision_counts = Counter(self.decisions)  # Example: {'yes': 5, 'no': 5}
        self.total_decisions = sum(self.decision_counts.values())  # Total number of decisions

        # Calculate key probabilities
        self.keys_probability = self._count_keys_probability()

    def run(self):
        entropy_results = {}

        # Overall entropy for all decisions
        entropy_results['entropy_value'] = self._entropy_calculation(**self.decision_counts)

        # Entropy for decisions by key
        for key, decision_list in self.data.items():
            decision_counts = Counter(decision_list)
            entropy_results[f'entropy_{key}'] = self._entropy_calculation(**decision_counts)

        # Conditional entropy calculation
        entropy_results['conditional_entropy'] = sum(
            (len(decision_list) / self.total_decisions) * entropy_results[f'entropy_{key}']
            for key, decision_list in self.data.items()
        )

        # Information gain, intrinsic information, and gain ratio
        entropy_results['information_gain'] = self._information_gain_calculation(
            entropy_results['entropy_value'], entropy_results['conditional_entropy']
        )
        entropy_results['intrinsic_info'] = self._entropy_calculation(**self.keys_probability)
        entropy_results['gain_ratio'] = self._gain_ratio_calculation(
            entropy_results['information_gain'], entropy_results['intrinsic_info']
        )

        # Display results
        print(entropy_results)

    def _entropy_calculation(self, **kwargs) -> float:
        """
        Calculate entropy given the counts of decision outcomes.
        :param kwargs: Example: yes=2, no=3
        :return: Entropy value
        """
        total = sum(kwargs.values())
        return -sum((count / total) * math.log2(count / total) for count in kwargs.values() if count > 0)

    def _information_gain_calculation(self, entropy_value: float, conditional_entropy_value: float) -> float:
        return entropy_value - conditional_entropy_value

    def _gain_ratio_calculation(self, information_gain_value: float, intrinsic_info_value: float) -> float:
        return information_gain_value / intrinsic_info_value if intrinsic_info_value != 0 else 0

    def _count_keys_probability(self):
        """
        Calculate the probability of each key based on the number of decisions.
        :return: A dictionary of key probabilities
        """
        return {key: len(decision_list) / self.total_decisions for key, decision_list in self.data.items()}

# Example usage
dec = DecisionTreeInductor()
dec.run()
