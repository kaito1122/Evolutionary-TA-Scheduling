"""
Kaito Minami
DS 3500 / Evolutionary TA Assignment
Assignment 4
Created: 03/19/2023 | Last Modified: 03/28/2023
GitHub Repo: https://github.khoury.northeastern.edu/minamik/evo_tas
"""

import random as rnd
import copy
from functools import reduce
import pandas as pd
import pickle
import time

class Evo:

    def __init__(self, groupname):
        self.pop = {}   # ((ob1, eval1), (obj2, eval2), ...) ==> solution
        self.objective = {}  # name -> objective func
        self.agents = {}   # name -> (agent operator, # input solutions)
        self.summary = pd.DataFrame(columns=['groupname'])
        self.groupname = groupname

    def size(self):
        """ The size of the solution population """
        return len(self.pop)

    def add_objective(self, name, f):
        """ Registering an objective with the Evo framework
        name - The name of the objective (string)
        f    - The objective function:   f(solution)--> a number """
        self.objective[name] = f
        self.summary[name] = None

    def add_agent(self, name, op, k=1):
        """ Registering an agent with the Evo framework
        name - The name of the agent
        op   - The operator - the function carried out by the agent  op(*solutions)-> new solution
        k    - the number of input solutions (usually 1) """
        self.agents[name] = (op, k)

    def get_random_solutions(self, k=1):
        """ Pick k random solutions from the population as a list of solutions
            We are returning DEEP copies of these solutions as a list """
        if self.size() == 0: # No solutions in the populations
            return []
        else:
            popvals = tuple(self.pop.values())
            return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]


    def add_solution(self, sol):
        """Add a new solution to the population """
        eval = tuple([(name, f(sol)) for name, f in self.objective.items()])
        self.pop[eval] = sol


    def run_agent(self, name):
        """ Invoke an agent against the current population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)


    def evolve(self, n=1, dom=100, status=100, sync=1000):
        """ To run n random agents against the population
        n - # of agent invocations
        dom - # of iterations between discarding the dominated solutions """

        agent_names = list(self.agents.keys())
        start_time = time.time()

        for i in range(n):
            if time.time() - start_time > 600:
                break

            pick = rnd.choice(agent_names) # pick an agent to run
            self.run_agent(pick)
            if i % dom == 0:
                self.remove_dominated()

            if i % status == 0: # print the population
                self.remove_dominated()
                print("Iteration: ", i)
                print("Population Size: ", self.size())
                print(self)

            if i % sync == 0: # synchronize
                try:
                    with open('solutions.dat', 'rb') as file:

                        # load saved population into a dictionary object
                        loaded = pickle.load(file)

                        # merge loaded solutions into my population
                        for eval, sol in loaded.items():
                            self.pop[eval] = sol
                except Exception as e:
                    print(e)

                # remove the dominated solutions
                self.remove_dominated()

                # resave the non-dominated solutions back to the file
                with open('solutions.dat', 'wb') as file:
                    pickle.dump(self.pop, file)

        # Clean up population
        self.remove_dominated()
        self.add_summary()

    @staticmethod
    def _dominates(p, q):
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p, q)}

    def remove_dominated(self):
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nds}


    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval,sol in self.pop.items():
            rslt += str(dict(eval))+":\t"+str(sol)+"\n"
        return rslt

    def add_summary(self):
        """
        Adds the Pareto-optimal results into DataFrame
        :return: content-added DataFrame
        """
        for eval in self.pop.keys():
            self.summary = pd.concat([self.summary, pd.Series(dict(eval)).to_frame().T], ignore_index=True)

        self.summary['groupname'] = [self.groupname] * self.summary.shape[0]

    def download(self):
        """
        Downloads Pareto-optimal solutions' DataFrame into CSV file
        :return: edited CSV file
        """
        # source: https://datatofish.com/export-dataframe-to-csv/
        self.summary.to_csv('downloads/solution.csv', index=False, header=True)