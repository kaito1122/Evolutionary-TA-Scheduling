"""
Kaito Minami
DS 3500 / Evolutionary TA Assignment
Assignment 4
Created: 03/19/2023 | Last Modified: 03/28/2023
GitHub Repo: https://github.khoury.northeastern.edu/minamik/evo_tas
"""

import pandas as pd
import pytest
import ta_section as tasec


@pytest.fixture
def test_cases():
    test1 = pd.read_csv('tests/test1.csv', header=None).to_numpy()
    test2 = pd.read_csv('tests/test2.csv', header=None).to_numpy()
    test3 = pd.read_csv('tests/test3.csv', header=None).to_numpy()
    return test1, test2, test3


def test_overallocation(test_cases):
    t1, t2, t3 = test_cases
    assert tasec.minimize_overallocation(t1) == 37, 'Minimize overallocation objective is not working'
    assert tasec.minimize_overallocation(t2) == 41, 'Minimize overallocation objective is not working'
    assert tasec.minimize_overallocation(t3) == 23, 'Minimize overallocation objective is not working'


def test_conflicts(test_cases):
    t1, t2, t3 = test_cases
    assert tasec.minimize_conflicts(t1) == 8, 'Minimize conflicts objective is not working'
    assert tasec.minimize_conflicts(t2) == 5, 'Minimize conflicts objective is not working'
    assert tasec.minimize_conflicts(t3) == 2, 'Minimize conflicts objective is not working'


def test_undersupport(test_cases):
    t1, t2, t3 = test_cases
    assert tasec.minimize_undersupport(t1) == 1, 'Minimize undersupport objective is not working'
    assert tasec.minimize_undersupport(t2) == 0, 'Minimize undersupport objective is not working'
    assert tasec.minimize_undersupport(t3) == 7, 'Minimize undersupport objective is not working'


def test_unwilling(test_cases):
    t1, t2, t3 = test_cases
    assert tasec.minimize_unwilling(t1) == 53, 'Minimize unwilling objective is not working'
    assert tasec.minimize_unwilling(t2) == 58, 'Minimize unwilling objective is not working'
    assert tasec.minimize_unwilling(t3) == 43, 'Minimize unwilling objective is not working'


def test_unpreferred(test_cases):
    t1, t2, t3 = test_cases
    assert tasec.minimize_unpreferred(t1) == 15, 'Minimize unpreferred objective is not working'
    assert tasec.minimize_unpreferred(t2) == 19, 'Minimize unpreferred objective is not working'
    assert tasec.minimize_unpreferred(t3) == 10, 'Minimize unpreferred objective is not working'
