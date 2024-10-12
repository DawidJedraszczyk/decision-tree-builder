import math
from collections import Counter
import pandas as pd
import graphviz
import pydotplus


class DecisionTreeInductor:
    def __init__(self, data: dict = None, target_attribute: str = 'Survived'):
        self.data = data
        self.decisions = []
        self.keys = []

        for decision_list in self.data.values():
            self.decisions.extend(decision_list)

        for key in self.data.keys():
            self.keys.append(key)

        self.decision_counts = Counter(self.decisions)
        self.total_decisions = sum(self.decision_counts.values())
        self.keys_probability = self._count_keys_probability()
        self.target_attribute = target_attribute

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

        entropy_results['information_gain'] = self._information_gain_calculation(
            entropy_results['entropy_value'],
            entropy_results['conditional_entropy']
        )

        entropy_results['intrinsic_info'] = self._intrinsic_info_calculation(
            **{str(k): v for k, v in self.keys_probability.items()}
        )

        entropy_results['gain_ratio'] = self._gain_ratio_calculation(
            entropy_results['information_gain'],
            entropy_results['intrinsic_info']
        )

        return entropy_results

    def build_tree(self, data, attributes, target_attribute):
        # Check if all decisions are the same
        decisions = data[target_attribute]
        if len(set(decisions)) == 1:
            return decisions.iloc[0]  # Return the decision (leaf node)

        # If no more attributes to split on, return the majority class
        if len(attributes) == 0:
            return decisions.mode()[0]  # Majority class

        # Find the best attribute by gain ratio
        results = {}
        for attribute in attributes:
            data_for_attribute = data[[attribute, target_attribute]]
            grouped_data = data_for_attribute.groupby(attribute)[target_attribute].apply(list).to_dict()
            dti = DecisionTreeInductor(data=grouped_data)
            result = dti.run()
            results[attribute] = result

        # Select the attribute with the highest gain ratio
        sorted_data = dict(sorted(results.items(), key=lambda item: item[1]['gain_ratio'], reverse=True))
        best_attribute = next(iter(sorted_data))

        # Create a node with the best attribute
        tree = {best_attribute: {}}

        # Split the data based on the best attribute and build the subtree recursively
        attribute_values = data[best_attribute].unique()
        for value in attribute_values:
            subset_data = data[data[best_attribute] == value]
            subtree = self.build_tree(subset_data, [attr for attr in attributes if attr != best_attribute],
                                      target_attribute)
            tree[best_attribute][value] = subtree

        return tree

    def _entropy_calculation(self, **kwargs) -> float:
        all_items = sum(value for value in kwargs.values())
        return -sum(value / all_items * math.log2(value / all_items) for value in kwargs.values())

    def _conditional_entropy_calculation(self, *args) -> float:
        return sum(probability * entropy for probability, entropy in args)

    def _information_gain_calculation(self, entropy_calculation_value: float,
                                      conditional_entropy_calculation_value: float) -> float:
        return entropy_calculation_value - conditional_entropy_calculation_value

    def _intrinsic_info_calculation(self, **kwargs):
        return self._entropy_calculation(**kwargs)

    def _gain_ratio_calculation(self, information_gain_calculation_value: float,
                                intrinsic_info_calculation_value: float) -> float:
        try:
            return information_gain_calculation_value / intrinsic_info_calculation_value
        except ZeroDivisionError:
            return 0

    def _count_keys_probability(self):
        return {key: len(decision_list) / self.total_decisions for key, decision_list in self.data.items()}

def visualize_tree(tree, parent_name='', graph=None):
    if graph is None:
        graph = graphviz.Digraph(format='png')

    for node, branches in tree.items():
        node_name = node
        graph.node(node_name)

        if parent_name:
            graph.edge(parent_name, node_name)

        if isinstance(branches, dict):
            # Recursively draw branches
            for branch, subtree in branches.items():
                branch_name = f"{node_name}_{branch}"
                graph.node(branch_name, label=str(branch))
                graph.edge(node_name, branch_name)

                if isinstance(subtree, dict):
                    visualize_tree(subtree, branch_name, graph)
                else:
                    # Leaf node
                    leaf_name = f"{branch_name}_leaf"
                    graph.node(leaf_name, label=str(subtree), shape='box')
                    graph.edge(branch_name, leaf_name)

    return graph

if __name__ == "__main__":
    data = pd.read_csv("data/titanic-homework.csv")
    attributes = [col for col in data.columns if col not in ['Survived', 'PassengerId', 'Name']]

    # Create the DecisionTreeInductor with data
    grouped_data = data.groupby(attributes[0])['Survived'].apply(list).to_dict()
    dti = DecisionTreeInductor(data=grouped_data)

    decision_tree = dti.build_tree(data, attributes, 'Survived')

    graph = visualize_tree(decision_tree)
    graph.render('decision_tree', view=True)












