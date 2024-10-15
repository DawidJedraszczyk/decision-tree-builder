import math
from collections import Counter
import pandas as pd
pd.options.mode.chained_assignment = None


class DecisionTreeInductor:
    def __init__(self, data: dict = None, target_label: str = 'Survived',  excluded_keys: list = []):
        self.data = data
        self.excluded_keys = excluded_keys
        self.TARGET_LABEL = target_label

        self.decisions = []

        self.keys = []
        self.key_lists = {}

        for key in self.data[0].keys():
            if key not in self.excluded_keys and key != self.TARGET_LABEL:
                self.keys.append(key)
                self.key_lists[key] = []

        for record in self.data:
            self.decisions.append(record[self.TARGET_LABEL])
            for key in self.keys:
                self.key_lists[key].append({'value': record[key], 'decision': record[self.TARGET_LABEL]})


        self.decision_counts = Counter(self.decisions)
        self.total_decisions = self.total_decisions = sum(decision for decision in self.decision_counts.values())  # 10
        self.keys_probabilities = self._count_keys_probabilities()

    def run(self):
        entropy_results = {}

        # entropy for all decisions
        entropy_results['entropy_value'] = self._entropy_calculation(
            **{str(k): v for k, v in self.decision_counts.items()})

        # entropy for decisions for one key
        conditional_entropy_data_holder = []
        for key, value_list in self.key_lists.items():
            # Group by key value, then count the decisions for each key value
            key_value_counts = {}
            for item in value_list:
                key_value = item['value']
                decision = item['decision']
                if key_value not in key_value_counts:
                    key_value_counts[key_value] = Counter()
                key_value_counts[key_value][decision] += 1

            # Calculate entropy for each key's value's decision counts
            key_conditional_entropy = 0
            intrinsic_info_data = {}
            for key_value, decision_counts in key_value_counts.items():
                # Entropy calculation for each key value
                entropy_value = self._entropy_calculation(**{str(k): v for k, v in decision_counts.items()})
                entropy_results[f'entropy_{key}_{key_value}'] = entropy_value

                # Probability of this key value
                probability = sum(decision_counts.values()) / self.total_decisions
                key_conditional_entropy += probability * entropy_value

                # Store probability for intrinsic info calculation
                intrinsic_info_data[key_value] = probability

                # Store data for conditional entropy calculation
                conditional_entropy_data_holder.append([probability, entropy_value])

            # Store the conditional entropy for this key
            entropy_results[f'conditional_entropy_{key}'] = key_conditional_entropy

            # Calculate Information Gain for this key
            information_gain = self._information_gain_calculation(
                entropy_results['entropy_value'], entropy_results[f'conditional_entropy_{key}']
            )
            entropy_results[f'information_gain_{key}'] = information_gain

            # Calculate Intrinsic Info for this key
            intrinsic_info = self._intrinsic_info_calculation(**{str(k): v for k, v in intrinsic_info_data.items()})
            entropy_results[f'intrinsic_info_{key}'] = intrinsic_info

            # Calculate Gain Ratio for this key
            gain_ratio = self._gain_ratio_calculation(information_gain, intrinsic_info)
            entropy_results[f'gain_ratio_{key}'] = gain_ratio


        return entropy_results

    def _entropy_calculation(self, **kwargs) -> float:
        """
        :param kwargs
        ex. yes= 2, no=2, maybe=3
        :return entropy value -> float
        ex. 1.557
        """
        all_items = sum(value for value in kwargs.values())
        return -sum(value / all_items * math.log2(value / all_items) for value in kwargs.values())

    def _conditional_entropy_calculation(self, *args) -> float:
        """
        :param args as lists
        ex. [0.2, 0], [0.2, 1], [0.6, 0.918]
        :return conditional entropy value -> float
        ex. 0.7508
        """
        return sum(probability * entropy for probability, entropy in args)

    def _information_gain_calculation(self, entropy_calculation_value: float, conditional_entropy_calculation_value: float) -> float:
        """
        :param args -> result of entropy_calculation -> float, conditional_entropy_calculation -> float:
        :return information gain value -> float
        """
        return entropy_calculation_value - conditional_entropy_calculation_value

    def _intrinsic_info_calculation(self, **kwargs):
        return self._entropy_calculation(**kwargs)

    def _gain_ratio_calculation(self, information_gain_calculation_value: float, intrinsic_info_calculation_value: float) -> float:
        return information_gain_calculation_value / intrinsic_info_calculation_value

    def _count_keys_probabilities(self):
        return {key: len(key_list) / self.total_decisions for key, key_list in self.key_lists.items()}

    def build_tree(self):
        pass


def categorize_age(age):
    if 0 <= age <= 20:
        return "young"
    elif 20 < age <= 40:
        return "middle"
    elif 40 < age <= 100:
        return "old"
    else:
        return "unknown"


if __name__ == "__main__":
    #data = pd.read_csv("data/titanic-homework.csv").to_dict(orient='records')

    data = [
        {'buying_price': 'high', 'doors':'4', 'safety': 'low', 'Survived': False},
        {'buying_price': 'high', 'doors':'5more', 'safety': 'high', 'Survived': True},
        {'buying_price': 'low', 'doors':'5more', 'safety': 'high', 'Survived': True},
        {'buying_price': 'low', 'doors':'5more', 'safety': 'low', 'Survived': False},
        {'buying_price': 'low', 'doors':'4', 'safety': 'med', 'Survived': True},
        {'buying_price': 'low', 'doors':'4', 'safety': 'high', 'Survived': True},
        {'buying_price': 'med', 'doors':'4', 'safety': 'high', 'Survived': False},
        {'buying_price': 'med', 'doors':'3', 'safety': 'high', 'Survived': True},
        {'buying_price': 'vhigh', 'doors':'3', 'safety': 'med', 'Survived': False},
        {'buying_price': 'vhigh', 'doors':'5more', 'safety': 'high', 'Survived': False},
    ]

    dti = DecisionTreeInductor(data=data)
    entropies = dti.run()
    print(entropies)