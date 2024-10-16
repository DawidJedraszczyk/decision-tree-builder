import math
from collections import Counter
import pandas as pd
pd.options.mode.chained_assignment = None

class DecisionTreeInductor:
    def __init__(self, data: dict = None, target_label: str = 'Survived',  excluded_keys: list = []):
        self.data = data
        self.excluded_keys = excluded_keys
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

    def build_tree(self, data=None, excluded_keys=None):
        """
        Recursively builds the decision tree.
        :param data: subset of data to build the tree on
        :param excluded_keys: list of keys to exclude from the splitting process (used in recursion)
        :return: decision tree (dict)
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
            return decisions[0]  # return the single decision

        # If no more attributes to split on, return the majority decision as a leaf node
        if len(excluded_keys) == len(self.keys):
            return decision_counts.most_common(1)[0][0]  # return the majority class

        # Select the best key to split on (you can choose either information gain or gain ratio)
        best_key = self._select_best_key(data, excluded_keys)

        # Create a node for the best key
        tree = {best_key: {}}

        # Add the best key to the excluded list so it isn't split again
        new_excluded_keys = excluded_keys.copy()
        new_excluded_keys.append(best_key)

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
        entropies = self.run()

        # Select the key with the highest gain ratio (you could choose information gain here instead)
        best_key = max((key for key in self.keys if key not in excluded_keys),
                       key=lambda k: entropies['gain_ratio'][k])

        return best_key

    def run(self):
        entropy_results = {
            'entropy': {},
            'conditional_entropy': {},
            'intrinsic_info': {},
            'information_gain': {},
            'gain_ratio': {}
        }

        # entropy for all decisions
        entropy_results['entropy']['value'] = self._entropy_calculation(
            **{str(k): v for k, v in self.decision_counts.items()})

        # entropy for decisions for one key
        conditional_entropy_data_holder = []
        for key, value_list in self.key_lists.items():

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
        return round(gain_value / intrinsic_info_calculation_value, 4)

    def _count_keys_probabilities(self):
        return {key: len(key_list) / self.total_decisions for key, key_list in self.key_lists.items()}


def categorize_age(age):
    if 0 <= age <= 20:
        return "young"
    elif 20 < age <= 40:
        return "middle"
    elif 40 < age <= 100:
        return "old"
    else:
        return "unknown"


def print_tree(tree, depth=0):
    """
    Recursively prints the decision tree in a readable format with indentation.
    :param tree: the decision tree (dict)
    :param depth: current depth level in the tree (used for indentation)
    """
    indent = "  " * depth  # Indentation based on the tree depth
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}{key}:")
            if isinstance(value, dict):
                print_tree(value, depth + 1)  # Recurse into subtree with increased depth
            else:
                print(f"{indent}  -> {value}")  # Print leaf node (decision)
    else:
        print(f"{indent}-> {tree}")  # Handle case for direct decision values


if __name__ == "__main__":
    data = pd.read_csv("data/titanic-homework.csv").to_dict(orient='records')

    # data = [
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

    dti = DecisionTreeInductor(data=data)#, target_label='Survived', excluded_keys=['PassengerId', 'Name', ])
    entropies = dti.build_tree()
    tree = dti.build_tree()

    print_tree(tree)