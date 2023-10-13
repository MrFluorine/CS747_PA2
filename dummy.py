import argparse
import pulp
from pulp import *
import numpy as np

parser = argparse.ArgumentParser()


class Solution:
    def __init__(self, path, alg):
        self.path = path
        print(self.get_mdp())

    def get_mdp(self):
        mdp = dict()
        data = open(self.path).read().strip().split("\n")
        for line in data:
            label, *content = line.split()
            if label == "numStates":
                mdp["ns"] = int(content[-1])
            elif label == "numActions":
                mdp["na"] = int(content[-1])
                mdp["T"] = np.zeros((mdp["ns"], mdp["na"], mdp["ns"]))
                mdp["R"] = np.zeros((mdp["ns"], mdp["na"], mdp["ns"]))
            elif label == "end":
                mdp["end"] = list(map(int, content))
            elif label == "transition":
                s, a, s_next, r, p = map(eval, content)
                mdp["R"][s, a, s_next], mdp["T"][s, a, s_next] = r, p
            elif label == "mdptype":
                mdp["mdptype"] = content[-1]
            elif label == "discount":
                mdp["gamma"] = float(content[-1])
        return mdp


if __name__ == "__main__":
    parser.add_argument("--mdp", type=str, default="./data/mdp/continuing-mdp-2-2.txt")
    parser.add_argument("--algorithm", type=str, default="vi")
    args = parser.parse_args()

    Solution(args.mdp, args.algorithm)
