import argparse
import pulp as p
import numpy as np

parser = argparse.ArgumentParser()


class Solution:
    def __init__(self, path, alg, policy_path):
        self.policy_path = policy_path
        self.path = path
        mdp, ini_policy = self.get_mdp()
        Value, policy = self.solver(alg, mdp, ini_policy)
        self.printing(Value, policy)

    def get_mdp(self):
        mdp = dict()
        data2 =None
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
        if self.policy_path:
            data2 = open(self.policy_path).read().strip().split("\n")
            data2 = [int(line) for line in data2]
            
            

        return mdp, data2

    def solver(self, alg, mdp, ini_policy):
        if alg == "vi":
            if ini_policy:
                ini_policy = np.array(ini_policy)
                mdp["R"]  = mdp["R"][np.arange(mdp["ns"]), ini_policy]
                mdp["T"]  = mdp["T"][np.arange(mdp["ns"]), ini_policy] 
                V = np.squeeze(np.linalg.inv(np.eye(mdp["ns"]) - mdp["gamma"] * mdp["T"])
                                @ np.sum(mdp["T"] * mdp["R"], axis=-1, keepdims=True))
                return V, ini_policy
            V = np.zeros(mdp["ns"])
            V_prev = V
            while True:
                V = np.max(np.sum(mdp["T"] * (mdp["R"] + mdp["gamma"] * V_prev), axis=2), axis=1)
                if np.allclose(V, V_prev, rtol=0, atol=1e-9):
                    break
                V_prev = V
            policy = np.argmax(np.sum(mdp["T"] * (mdp["R"] + mdp["gamma"] * V), axis=2), axis=1)
            return V, policy
        if alg == "hpi":
            if ini_policy:
                ini_policy = np.array(ini_policy)
                mdp["R"]  = mdp["R"][np.arange(mdp["ns"]), ini_policy]
                mdp["T"]  = mdp["T"][np.arange(mdp["ns"]), ini_policy] 
                V = np.squeeze(np.linalg.inv(np.eye(mdp["ns"]) - mdp["gamma"] * mdp["T"])
                                @ np.sum(mdp["T"] * mdp["R"], axis=-1, keepdims=True))
                return V, ini_policy
            policy = np.random.randint(low=0, high=mdp["na"], size=mdp["ns"])
            policy_old = policy
            while True:
                R_policy = mdp["R"][np.arange(mdp["ns"]), policy_old]
                T_policy = mdp["T"][np.arange(mdp["ns"]), policy_old]
                V = np.squeeze(np.linalg.inv(np.eye(mdp["ns"]) - mdp["gamma"] * T_policy)
                               @ np.sum(T_policy * R_policy, axis=-1, keepdims=True))

                policy = np.argmax(np.sum(mdp["T"] * (mdp["R"] + mdp["gamma"] * V), axis=2), axis=1)
                if np.array_equal(policy, policy_old):
                    break

                policy_old = policy
            return V, policy
        if alg == "lp":
            if ini_policy:
                ini_policy = np.array(ini_policy)
                mdp["R"]  = mdp["R"][np.arange(mdp["ns"]), ini_policy]
                mdp["T"]  = mdp["T"][np.arange(mdp["ns"]), ini_policy] 
                V = np.squeeze(np.linalg.inv(np.eye(mdp["ns"]) - mdp["gamma"] * mdp["T"])
                                @ np.sum(mdp["T"] * mdp["R"], axis=-1, keepdims=True))
                return V, ini_policy

            problem = p.LpProblem('MDP', p.LpMinimize)
            V = np.array(list(p.LpVariable.dicts("V", [i for i in range(mdp["ns"])]).values()))
            problem += p.lpSum(V) 

            for state in range(mdp["ns"]):
                for action in range(mdp["na"]):
                    problem += V[state] >= p.lpSum(mdp["T"][state, action] * (mdp["R"][state, action] + mdp["gamma"] * V))
            problem.solve(p.apis.PULP_CBC_CMD(msg=0))
            V = np.array(list(map(p.value, V)))
            policy = np.argmax(np.sum(mdp["T"] * (mdp["R"] + mdp["gamma"] * V), axis=-1), axis=-1)
            return V, policy

    def printing(self, value, policy):
        value = list(map('{0:.6f}'.format, list(value)))
        policy = list(map(str, policy))
        output = ""
        for i in range(len(value)):
            output += value[i] + " " + policy[i] + "\n"
        print(output.strip())


if __name__ == "__main__":
    parser.add_argument("--mdp", type=str, default="./data/mdp/continuing-mdp-2-2.txt")
    parser.add_argument("--algorithm", type=str, default="lp")
    parser.add_argument("--policy", type=str, default=None)
    args = parser.parse_args()

    Solution(args.mdp, args.algorithm, args.policy)
