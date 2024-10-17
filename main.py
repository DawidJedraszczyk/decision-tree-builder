import math
from collections import Counter
import pandas as pd
import graphviz
pd.options.mode.chained_assignment = None


class DecisionTreeInductor:
    def __init__(self, data: dict = None, target_label: str = 'Survived',  excluded_keys: set = None):
        self.data = data
        self.excluded_keys = excluded_keys or set()
        self.TARGET_LABEL = target_label


        self.decisions = [record[self.TARGET_LABEL] for record in data]
        self.decision_counts = Counter(self.decisions)

        self.keys = []
        self.key_lists = {}

        for key in self.data[0].keys():
            if key not in self.excluded_keys and key != self.TARGET_LABEL:
                self.keys.append(key)
                self.key_lists[key] = []

        for record in self.data:
            for key in self.keys:
                self.key_lists[key].append({'value': record[key], 'decision': record[self.TARGET_LABEL]})


        self.total_decisions = self.total_decisions = sum(decision for decision in self.decision_counts.values())  # 10
        self.keys_probabilities = self._count_keys_probabilities()
        self.graph = graphviz.Digraph(comment='Decision Tree')


    def build_tree(self, data: list = None, excluded_keys: set=None):
        """
        Recursively builds the decision tree.
        :param data: subset of data to build the tree on
        :param excluded_keys: set of keys to exclude from the splitting process (used in recursion)
        :return: decision tree (dict) or leaf node with decision and class counts
        """
        if data is None:
            data = self.data

        if excluded_keys is None:
            excluded_keys = self.excluded_keys.copy()

        # Get the counts of each decision in the current data subset
        decisions = [record[self.TARGET_LABEL] for record in data]
        decision_counts = Counter(decisions)

        # If all the data has the same decision (pure node), return it as a leaf node
        if len(decision_counts) == 1:
            return decisions[0], dict(decision_counts)  # return the single decision with the class counts

        # If no more attributes to split on, return the majority decision as a leaf node
        ended = True
        for key in self.keys:
            if key not in excluded_keys:
                ended = False
                break
        if ended:
            return decision_counts.most_common(1)[0][0], dict(decision_counts)  # return the majority class with counts

        # Select the best key to split on (you can choose either information gain or gain ratio)

        best_key = self._select_best_key(data, excluded_keys)
        # Create a node for the best key
        tree = {best_key: {}}

        # Add the best key to the excluded list so it isn't split again
        new_excluded_keys = excluded_keys.copy()
        new_excluded_keys.add(best_key)

        # Get the unique values of the best key and build branches for each
        key_values = set([record[best_key] for record in data])

        for value in key_values:
            # Filter the data for this branch
            branch_data = [record for record in data if record[best_key] == value]

            # Recursively build the subtree for this branch
            subtree = self.build_tree(branch_data, new_excluded_keys)
            tree[best_key][value] = subtree

        return tree

    def _select_best_key(self, data, excluded_keys):
        """
        Select the best key (attribute) to split the data on, based on Information Gain or Gain Ratio.
        :param data: current subset of the data
        :param excluded_keys: list of keys to exclude from splitting
        :return: the key with the highest information gain or gain ratio
        """
        # Calculate information gain or gain ratio for each key
        entropies = self.run(data)
        print(entropies)

        # Select the key with the highest gain ratio (you could choose information gain here instead)
        best_key = max((key for key in self.keys if key not in excluded_keys),
                       key=lambda k: entropies['gain_ratio'][k])
        print("Wyznaczono: ", best_key)
        print("")

        return best_key

    def run(self, data=None):
        """
        Calculates entropy, conditional entropy, information gain, etc. for a given dataset.
        If no data is provided, it defaults to the entire dataset.
        """
        if data is None:
            data = self.data

        entropy_results = {
            'entropy': {},
            'conditional_entropy': {},
            'intrinsic_info': {},
            'information_gain': {},
            'gain_ratio': {}
        }

        # Calculate entropy for the current subset of data
        decisions = [record[self.TARGET_LABEL] for record in data]
        decision_counts = Counter(decisions)

        entropy_results['entropy']['value'] = self._entropy_calculation(
            **{str(k): v for k, v in decision_counts.items()}
        )

        # Calculate entropy and other metrics for each key
        key_lists = {key: [] for key in self.keys if key not in self.excluded_keys}
        for record in data:
            for key in key_lists:
                key_lists[key].append({'value': record[key], 'decision': record[self.TARGET_LABEL]})

        conditional_entropy_data_holder = []
        for key, value_list in key_lists.items():
            entropy_results['entropy'][key] = {}
            entropy_results['conditional_entropy'][key] = {}
            entropy_results['intrinsic_info'][key] = {}
            entropy_results['information_gain'][key] = {}
            entropy_results['gain_ratio'][key] = {}

            # Group by key value, then count the decisions for each key value
            key_value_counts = {}
            for item in value_list:
                key_value = item['value']
                decision = item['decision']
                if key_value not in key_value_counts:
                    key_value_counts[key_value] = Counter()
                key_value_counts[key_value][decision] += 1

            key_conditional_entropy = 0
            intrinsic_info_data = {}

            for key_value, decision_counts in key_value_counts.items():

                # Entropy calculation for each key value
                entropy_value = self._entropy_calculation(**{str(k): v for k, v in decision_counts.items()})
                entropy_results['entropy'][key][key_value] = entropy_value

                # Probability of this key value
                probability = sum(decision_counts.values()) / self.total_decisions
                key_conditional_entropy += probability * entropy_value

                # Store probability for intrinsic info calculation
                intrinsic_info_data[key_value] = probability

                # Store data for conditional entropy calculation
                conditional_entropy_data_holder.append([probability, entropy_value])


            # Store the conditional entropy for this key
            entropy_results['conditional_entropy'][key] = key_conditional_entropy

            # Calculate Information Gain for this key
            information_gain = self._information_gain_calculation(
                entropy_results['entropy']['value'], entropy_results['conditional_entropy'][key]
            )
            entropy_results['information_gain'][key] = information_gain

            # Calculate Intrinsic Info for this key
            intrinsic_info = self._intrinsic_info_calculation(**{str(k): v for k, v in intrinsic_info_data.items()})
            entropy_results['intrinsic_info'][key] = intrinsic_info

            # Calculate Gain Ratio for this key
            gain_value = entropy_results['entropy']['value'] - key_conditional_entropy
            gain_ratio = self._gain_ratio_calculation(gain_value, intrinsic_info)
            entropy_results['gain_ratio'][key] = gain_ratio

        return entropy_results

    def visualize_tree(self, tree=None, parent_id='root'):
        """
        Recursively generates a Graphviz visualization of the decision tree with keys as sub-nodes and values as leafs.
        :param tree: the decision tree (dict or tuple)
        :param parent_id: the id of the parent node
        """
        if tree is None:
            tree = self.build_tree()  # Build the tree if not provided

        if isinstance(tree, dict):
            # Internal node (Key)
            key = next(iter(tree))  # Get the key (attribute)
            key_node_id = f"{parent_id}_{key}"  # Create a unique node ID for the key
            self.graph.node(key_node_id, key, shape='ellipse')  # Create a node for the key

            if parent_id != 'root':
                self.graph.edge(parent_id, key_node_id)  # Connect the current node to its parent

            for value, subtree in tree[key].items():
                # Value node (each value of the key as an edge)
                value_node_id = f"{key_node_id}_{value}"  # Create a unique node ID for the value
                self.graph.node(value_node_id, str(value), shape='box')  # Create a node for the value

                # Connect key to the value
                self.graph.edge(key_node_id, value_node_id, label=str(value))

                # Recursively visualize the subtree
                self.visualize_tree(subtree, parent_id=value_node_id)

        elif isinstance(tree, tuple):
            # Leaf node: show decision and class counts
            decision, class_counts = tree
            leaf_label = f"Decision: {decision}\n" + ", ".join([f"Count: {cnt}" for cls, cnt in class_counts.items()])
            self.graph.node(parent_id, leaf_label, shape='box')  # Leaf node in box shape

        else:
            # Handle direct leaf case (if it's just a decision, not a tuple)
            self.graph.node(parent_id, f"Decision: {tree}", shape='box')

    def render_tree(self):
        """Render and display the decision tree as an image."""
        self.graph.render('decision_tree', format='png', view=True)  # Render the tree



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

    def _entropy_calculation(self, **kwargs) -> float:
        """
        :param kwargs
        ex. yes= 2, no=2, maybe=3
        :return entropy value -> float
        ex. 1.557
        """
        all_items = sum(value for value in kwargs.values())
        return -sum(value / all_items * math.log2(value / all_items) for value in kwargs.values() if value != 0)

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

    def _gain_ratio_calculation(self, gain_value: float, intrinsic_info_calculation_value: float) -> float:
        return round(gain_value / intrinsic_info_calculation_value, 4) if intrinsic_info_calculation_value != 0 else 0

    def _count_keys_probabilities(self):
        return {key: len(key_list) / self.total_decisions for key, key_list in self.key_lists.items()}

    def print_tree(self, tree, depth=0):
        """
        Recursively prints the decision tree in a readable format with indentation.
        :param tree: the decision tree (dict or tuple)
        :param depth: current depth level in the tree (used for indentation)
        """
        indent = "  " * depth  # Indentation based on the tree depth
        if isinstance(tree, dict):
            for key, value in tree.items():
                print(f"{indent}{key}:")
                if isinstance(value, dict):
                    self.print_tree(value, depth + 1)  # Recurse into subtree with increased depth
                else:
                    self.print_tree(value, depth + 1)  # Handle the case when leaf node contains a tuple
        elif isinstance(tree, tuple):
            # Leaf node: unpack the decision and the class counts
            decision, class_counts = tree
            counts_str = ', '.join([f"{cls}: {count}" for cls, count in class_counts.items()])
            print(f"{indent}-> {decision} (Counts: {counts_str})")
        else:
            print(f"{indent}-> {tree}")  # Handle case for direct decision values

    def check_prediction(self, tree, record):
        """
        Checks the prediction for a single record in the decision tree.
        :param tree: Decision tree (dict).
        :param record: Single record (dict) from the dataset.
        :return: Predicted value (e.g., True/False for target label).
        """
        if isinstance(tree, dict):
            attribute = list(tree.keys())[0]
            value = record[attribute]

            if value in tree[attribute]:
                return self.check_prediction(tree[attribute][value], record)
            else:
                return None
        elif isinstance(tree, tuple):
            return tree[0]
        else:
            return tree



    def evaluate_tree(self, tree: dict, data:list) -> float:
        """
        Checks every single row of data if it match the tree
        :param tree: dict.
        :param data: data -> list of dicts.
        :return: Wynik procentowy poprawnych przewidywań.
        """
        correct_predictions = 0
        total_records = len(data)

        for record in data:
            # Pobierz przewidywanie z drzewa
            prediction = self.check_prediction(tree, record)

            # Sprawdź, czy przewidywanie zgadza się z rzeczywistą wartością
            if prediction is not None and prediction == record[self.TARGET_LABEL]:
                correct_predictions += 1

        accuracy = correct_predictions / total_records * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_records} correct predictions)")

        return accuracy


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
    data = pd.read_csv("data/titanic-homework.csv").to_dict(orient='records')
    #data = pd.read_csv("data/Breast_Cancer.csv").to_dict(orient='records')

    # dataset_from_pres = [
    #     {'buying_price': 'high', 'doors':'4', 'safety': 'low', 'Survived': False},
    #     {'buying_price': 'high', 'doors':'5more', 'safety': 'high', 'Survived': True},
    #     {'buying_price': 'low', 'doors':'5more', 'safety': 'high', 'Survived': True},
    #     {'buying_price': 'low', 'doors':'5more', 'safety': 'low', 'Survived': False},
    #     {'buying_price': 'low', 'doors':'4', 'safety': 'med', 'Survived': True},
    #     {'buying_price': 'low', 'doors':'4', 'safety': 'high', 'Survived': True},
    #     {'buying_price': 'med', 'doors':'4', 'safety': 'high', 'Survived': False},
    #     {'buying_price': 'med', 'doors':'3', 'safety': 'high', 'Survived': True},
    #     {'buying_price': 'vhigh', 'doors':'3', 'safety': 'med', 'Survived': False},
    #     {'buying_price': 'vhigh', 'doors':'5more', 'safety': 'high', 'Survived': False},
    # ]


    for record in data:
        record['Age'] = categorize_age(record['Age'])

    #ages = [{'age': values['Age'], 'decision': values['Survived']} for values in data]
    #ages_sorted = sorted(ages, key=lambda x: x['age'])


    dti = DecisionTreeInductor(data=data, target_label='Survived', excluded_keys={'PassengerId', 'Name'} )
    tree = dti.build_tree()
    dti.print_tree(tree)
    dti.visualize_tree(tree)
    dti.render_tree()  # Renders the tree to a file and opens the image
    dti.evaluate_tree(tree, data)
