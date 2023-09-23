import itertools

def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4  

def test_wrong_answer():
    assert not inc(3) == 5      


def test_hparam_combinations_count():
    # a test to check that all possible cobminatons of params
    gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
    C_range = [0.1, 1.0, 2, 5, 10]
    hparam_combos = list(itertools.product(gamma_range, C_range))

    assert len(hparam_combos)==len(gamma_range)* len(C_range)

def test_hparam_combinations_value():
    # a test to check that all possible cobminatons of params
    gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
    C_range = [0.1, 1.0, 2, 5, 10]
    hparam_combos = list(itertools.product(gamma_range, C_range))

    expected_hparam_combo1 = (0.001, 1.0)

    assert expected_hparam_combo1 in hparam_combos  