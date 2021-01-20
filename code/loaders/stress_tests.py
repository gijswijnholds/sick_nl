from sick_nl.code.loaders.sick import load_sick_nl
from sick_nl.code.config import prep_order_fn, present_tense_fn


def load_stress_test(test_fn: str):
    with open(test_fn) as inf:
        data = [ln.strip().split('\t') for ln in inf.readlines()[1:]]
    return data


def load_prep_order():
    return load_stress_test(prep_order_fn)


def load_present_tense():
    return load_stress_test(present_tense_fn)


def load_switched_sick(stress_data):
    nl_sick = load_sick_nl()
    stress_dict = {s1: s2 for (s1, s2, el) in stress_data}
    before_data = list(set([(s1, s2, el, rl) for (s1, s2, el, rl)
                            in nl_sick.test_data if s1 in stress_dict] +
                           [(s1, s2, el, rl) for (s1, s2, el, rl)
                            in nl_sick.test_data if s2 in stress_dict]))
    after_data = list(set([(stress_dict[s1], s2, el, rl) for (s1, s2, el, rl)
                           in nl_sick.test_data if s1 in stress_dict and s2 not in stress_dict] +
                          [(s1, stress_dict[s2], el, rl) for (s1, s2, el, rl)
                           in nl_sick.test_data if s2 in stress_dict and s1 not in stress_dict] +
                          [(stress_dict[s1], stress_dict[s2], el, rl) for (s1, s2, el, rl)
                           in nl_sick.test_data if s2 in stress_dict and s1 in stress_dict]))
    return before_data, after_data


def load_stress_test_bidirectional(stress_data):
    left_to_right_data = [(s1, s2, el, 5.0) for (s1, s2, el) in stress_data]
    right_to_left_data = [(s2, s1, el, 5.0) for (s1, s2, el) in stress_data]
    return left_to_right_data, right_to_left_data
