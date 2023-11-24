"""
Kaito Minami
DS 3500 / Evolutionary TA Assignment
Assignment 4
Created: 03/19/2023 | Last Modified: 03/28/2023
GitHub Repo: https://github.khoury.northeastern.edu/minamik/evo_tas
"""

import pandas as pd
import random as rnd
import numpy as np
from evo import Evo

# load sections data
# ['section', 'instructor', 'daytime', 'location', 'students', 'topic', 'min_ta', 'max_ta']
sections = pd.read_csv('data/sections.csv')
sections_easy = pd.read_csv('data/sections_easy.csv')

# load tas data
#  ['ta_id', 'name', 'max_assigned',
#  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
# U = Unavailable, W = Willing, P = Preferred
tas = pd.read_csv('data/tas.csv')


def minimize_overallocation(P):
    """
    Returns the total number of overallocated sections (how many more than max_assigned)
    :param P: (NumPy array) binary TA-section assignment sheet
    :return: (int) the total number of allocated sections
    """
    return sum([x - y for x, y in zip([sum(z) for z in P], list(tas['max_assigned'])) if x > y])


def minimize_conflicts(P):
    """
    Returns the total number of TA who has time conflict
    :param P: (NumPy array) binary TA-section assignment sheet
    :return: (int) the total number of TA who has time conflict
    """
    # the section's assignment status and daytime zipped list
    daytime = [list(zip(x, sections['daytime'])) for x in P]

    # in the zipped file, filter only the assigned sections
    assigned = [list(filter(lambda x: 1 in x, item)) for item in daytime]

    # when not all the sections have unique time value, count it as 1 more time conflict
    return sum([1 for a in assigned if len(a) != len(set(a))])


def minimize_undersupport(P):
    """
    Returns the total number of sections with under-support (less than minimum TA requirement)
    :param P: (NumPy array) binary TA-section assignment sheet
    :return: (int) the total number of sections with under-support
    """
    return sum([(y - x) for x, y in zip([sum(t) for t in P.transpose()], sections['min_ta']) if x < y])


def minimize_unwilling(P):
    """
    Returns the total number of unwilling sections ('U' but assigned nevertheless)
    :param P: (NumPy array) binary TA-section assignment sheet
    :return: (int) the total number of unwilling sections
    """
    return sum([1 for t in range(P.shape[0]) for x in range(P.shape[1]) if P[t][x] == 1 and tas.loc[t, str(x)] == 'U'])


def minimize_unpreferred(P):
    """
    Returns the total number of unpreferred sections ('W' not 'P' but assigned nevertheless)
    :param P: (NumPy array) binary TA-section assignment sheet
    :return: (int) the total number of unpreferred sections
    """
    return sum([1 for t in range(P.shape[0]) for x in range(P.shape[1]) if P[t][x] == 1 and tas.loc[t, str(x)] == 'W'])


def rand_switcher(solutions):
    """
    Switches one TA-section between 0 and 1
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :return: (Numpy array) the first picked solution with a randomly switched TA-section
    """
    P = solutions[0]

    # i and j are randomized values by col and row respectively
    i = rnd.randrange(0, P.shape[0])
    j = rnd.randrange(0, P.shape[1])

    # if it's on, turns it off and vice versa
    if P[i, j] == 0:
        P[i, j] = 1
    elif P[i, j] == 1:
        P[i, j] = 0
    return P


def fix_over_under(solutions, T=False):
    """
    Fixes both over- and under-assignment by distributing 1 to correct P and W location
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :param T: (boolean) to fix by rows or columns (row=False, col=True)
    :return: (Numpy array) the first picked solution with fixed number of assignments
    """
    P = solutions[0]

    # test by rows or cols?
    if T:
        P = P.transpose()
        compare = sections['min_ta']
    else:
        compare = tas['max_assigned']

    # list of tuples of
    # over-, under-assigned sections' coordinates (only rows' or cols') and desired number of assignment
    over_under = [(i, y) for i, y in zip(range(len(P)), compare) if sum(P[i]) != y]

    if over_under:
        # preference table (composed of ['P', 'U', 'W'])
        pref = np.array(tas.loc[:, '0':'16'])

        for items in over_under:
            # among the problematic coordinate, filter through to get only the ['P', 'W']'s that is supposed to be 1
            # and shuffles to give randomness in case of over-assignment
            ones = [i for i in range(len(pref[items[0]])) if pref[items[0]][i] in ['P', 'W']]
            rnd.shuffle(ones)

            # limit the number of ones to prevent over- and under-assignment
            if len(ones) > items[1]:
                ones = ones[:items[1]]
            elif len(ones) < items[1]:
                ones += [rnd.randint(0, 16) for _ in range(items[1] - len(ones))]

            # assigns the ones' coordinates into reset rows (cols)
            reset = [0] * len(P[items[0]])
            for i in ones:
                reset[i] = 1
            P[items[0]] = reset

    if T:
        P = P.transpose()
    return P


def fix_overallocation(solutions):
    """
    Fixes the overallocation of TAs to sections (objective)
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :return: (Numpy array) the first picked solution with fixed number of assignments
    """
    return fix_over_under(solutions)


def fix_undersupport(solutions):
    """
    Fixes the undersupport of sections by TAs (objective)
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :return: (Numpy array) the first picked solution with fixed number of assignments
    """
    return fix_over_under(solutions, T=True)


def puw_switcher(solutions, fix):
    """
    Switches between 0 and 1 for specified assignment preference status ['P', 'U', 'W']
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :param fix: (str) specified assignment preference status ['P', 'U', 'W']
    :return: (Numpy array) the first picked solution with the specified status fixed
    """
    # preference table (composed of ['P', 'U', 'W'])
    pref = np.array(tas.loc[:, '0':'16'])

    # index coordinates of fix preference status in the preference table
    indices = np.nonzero(pref == fix)
    index = [(x, y) for x, y in zip(indices[0], indices[1])]

    P = solutions[0]
    for idx in range(len(index)):
        coords = index[idx]
        if fix == 'P':
            # turn on
            if P[coords[0]][coords[1]] == 0:
                P[coords[0]][coords[1]] = 1
        else:
            # turn off
            if P[coords[0]][coords[1]] == 1:
                P[coords[0]][coords[1]] = 0

    return P


def turn_p(solutions):
    """
    Turns all the 'P' TA-sections on
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :return: (Numpy array) the first picked solution with all the 'P' TA-section on
    """
    return puw_switcher(solutions, 'P')


def fix_u(solutions):
    """
    Turns all the 'U' TA-sections off
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :return: (Numpy array) the first picked solution with all the 'U' TA-section off
    """
    return puw_switcher(solutions, 'U')


def fix_w(solutions):
    """
    Turns all the 'W' TA-sections off
    :param solutions: (list of Numpy array) list of solutions (binary TA-section assignment sheet)
    :return: (Numpy array) the first picked solution with all the 'W' TA-section off
    """
    return puw_switcher(solutions, 'W')


def main():

    E = Evo(groupname='Kaito')

    E.add_objective('overallocation', minimize_overallocation)
    E.add_objective('conflicts', minimize_conflicts)
    E.add_objective('undersupport', minimize_undersupport)
    E.add_objective('unwilling', minimize_unwilling)
    E.add_objective('unpreferred', minimize_unpreferred)

    # fix agent
    E.add_agent("rand_switcher", rand_switcher, k=1)
    E.add_agent("fix_overallocation", fix_overallocation, k=1)
    E.add_agent("fix_undersupport", fix_undersupport, k=1)
    E.add_agent("turn_P", turn_p, k=1)
    E.add_agent("fix_U", fix_u, k=1)
    E.add_agent("fix_W", fix_w, k=1)

    # NumPy array of 0s and 1s (0 not-assigned, 1 assigned)
    # P = np.array([rnd.randrange(0,2) for _ in range(17*43)]).reshape(43,17)
    test1 = pd.read_csv('tests/test1.csv', header=None).to_numpy()
    test2 = pd.read_csv('tests/test2.csv', header=None).to_numpy()
    test3 = pd.read_csv('tests/test3.csv', header=None).to_numpy()

    E.add_solution(test1)
    # E.add_solution(test2)
    # E.add_solution(test3)
    print(E)

    E.evolve(100000000000, 100, 10000)
    print(E)
    E.download()


if __name__ == '__main__':
    main()
