import math

def entropy_calculation(**kwargs) -> float:
    """
    :param kwargs
    ex. yes= 2, no=2, maybe=3
    :return entropy value -> float
    ex. 1.557
    """
    all_items = sum(value for value in kwargs.values())
    return -sum(value/all_items * math.log2(value/all_items) for value in kwargs.values())


def conditional_entropy_calculation(*args) -> float:
    """
    :param args as lists
    ex. [0.2, 0], [0.2, 1], [0.6, 0.918]
    :return conditional entropy value -> float
    ex. 0.7508
    """
    return sum(probability * entropy for probability, entropy in args)


def information_gain_calculation(entropy_calculation_value : float, conditional_entropy_calculation_value : float) -> float:
    """
    :param args -> result of entropy_calculation -> float, conditional_entropy_calculation -> float:
    :return information gain value -> float
    """
    return entropy_calculation_value - conditional_entropy_calculation_value

def intrinsic_infro_calculation():
    #TODO
    pass

def gain_ratio_calculation():
    #TODO
    pass



