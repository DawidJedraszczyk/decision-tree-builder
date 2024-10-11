import math
from collections import Counter
import pandas as pd



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
        self.decisions = []  # no, no, yes, no, ...
        self.keys = []  # low, med, high

        for decision_list in self.data.values():
            self.decisions.extend(decision_list)

        for key in self.data.keys():
            self.keys.append(key)

        self.decision_counts = Counter(self.decisions)  # no = 5 yes = 5
        self.total_decisions = self.total_decisions = sum(decision for decision in self.decision_counts.values())  # 10
        self.keys_probability = self._count_keys_probability()
        print(self.keys_probability)

    def run(self):
        entropy_results = {}

        # entropy for all decisions
        entropy_results['entropy_value'] = self._entropy_calculation(
            **{str(k): v for k, v in self.decision_counts.items()})

        # entropy for decisions for one key
        for key, decision_list in self.data.items():
            decision_counts = Counter(decision_list)
            entropy_value = self._entropy_calculation(**{str(k): v for k, v in decision_counts.items()})

            # Dynamically create a key for each entropy result
            entropy_results[f'entropy_{key}'] = entropy_value

        conditional_entropy = 0
        for key, decision_list in self.data.items():
            decision_size = len(decision_list)
            probability = decision_size / self.total_decisions
            conditional_entropy += (probability * entropy_results[f'entropy_{key}'])

        entropy_results['conditional_entropy'] = conditional_entropy

        entropy_results['information_gain'] = self._information_gain_calculation(entropy_results['entropy_value'],
                                                                                 entropy_results['conditional_entropy'])

        entropy_results['intrinsic_info'] = self._intrinsic_info_calculation(**{str(k): v for k, v in self.keys_probability.items()})

        entropy_results['gain_ratio'] = self._gain_ratio_calculation(entropy_results['information_gain'],
                                                                     entropy_results['intrinsic_info'])

        print(entropy_results)
        print("")

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

    def _information_gain_calculation(self, entropy_calculation_value: float,
                                      conditional_entropy_calculation_value: float) -> float:
        """
        :param args -> result of entropy_calculation -> float, conditional_entropy_calculation -> float:
        :return information gain value -> float
        """
        return entropy_calculation_value - conditional_entropy_calculation_value

    def _intrinsic_info_calculation(self, **kwargs):
        return self._entropy_calculation(**kwargs)

    def _gain_ratio_calculation(self, information_gain_calculation_value: float,
                                intrinsic_info_calculation_value: float) -> float:
        return information_gain_calculation_value / intrinsic_info_calculation_value

    def _count_keys_probability(self):
        return {key: len(decision_list) / self.total_decisions for key, decision_list in self.data.items()}

if __name__ == "__main__":
    data = pd.read_csv("data/titanic-homework.csv")
    attributes = [col for col in data.columns if col not in ['Survived', 'PassengerId', 'Name']]

    for attribute in attributes:
        data_for_attribute = data[[attribute, 'Survived']]
        grouped_data = data_for_attribute.groupby(attribute)['Survived'].apply(list).to_dict()  # Convert to dict

        print(attribute)
        dti = DecisionTreeInductor(data=grouped_data)  # Pass as a dict
        dti.run()