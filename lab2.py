import math
from collections import Counter
import pandas as pd
from graphviz import Digraph
import pydotplus


class DecisionTreeNode:
    def __init__(self, attribute=None, value=None, branches=None, is_leaf=False, label=None):
        self.attribute = attribute  # Attribute the node splits on
        self.value = value  # Value of the attribute
        self.branches = branches or {}  # Dictionary of child nodes for each attribute value
        self.is_leaf = is_leaf  # True if it's a leaf node
        self.label = label  # Class label if it's a leaf node


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
        self.graph = Digraph()
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

    def build_tree(self, data, attributes):
        # Base cases
        if all_same_class(data['Survived']):
            return DecisionTreeNode(is_leaf=True, label=data['Survived'].iloc[0])

        if not attributes:
            # Return a leaf node with the most common class
            most_common_label = data['Survived'].mode()[0]
            return DecisionTreeNode(is_leaf=True, label=most_common_label)

        # Find the best attribute to split on
        best_attribute = self._find_best_attribute(data, attributes)

        # Create a new decision tree node for the best attribute
        node = DecisionTreeNode(attribute=best_attribute)

        # Split the dataset and recursively build the tree for each subset
        for value in data[best_attribute].unique():
            subset = data[data[best_attribute] == value]
            # Remove the current attribute from the set of available attributes
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            node.branches[value] = self.build_tree(subset, new_attributes)

        return node

    def _find_best_attribute(self, data, attributes):
        # Implement information gain or gain ratio to select the best attribute
        best_gain = -float('inf')
        best_attribute = None

        for attribute in attributes:
            grouped_data = data.groupby(attribute)['Survived'].apply(list).to_dict()
            self.data = grouped_data
            self.run()
            # Use information gain or gain ratio from the run results
            gain = self._information_gain_calculation(
                self._entropy_calculation(**{str(k): v for k, v in self.decision_counts.items()}),
                self._conditional_entropy_calculation(
                    (len(grouped_data[key]) / self.total_decisions,
                     self._entropy_calculation(**Counter(grouped_data[key]))) for key in grouped_data
                )
            )

            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute

        return best_attribute

    def visualize_tree(self):
        self.graph.render('decision_tree', view=True, format='png')

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

def all_same_class(column):
    """Check if all values in the column are the same (i.e., same class)."""
    return column.nunique() == 1

if __name__ == "__main__":
    data = pd.read_csv("data/titanic-homework.csv")
    attributes = [col for col in data.columns if col not in ['Survived', 'PassengerId', 'Name']]

    dti = DecisionTreeInductor()
    decision_tree = dti.build_tree(data, attributes)
    dti.visualize_tree()  # This will render and open the tree visualization