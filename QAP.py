from typing import Any

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float64)


class QAPDataset(Dataset):
    def __init__(
        self,
        size,
        num_instances,
        single_instance=False,
        sym_dist=True,
        sym_flow=True,
        tsp_ver=False,
        device="cpu",
        seed=None,
    ):
        if seed is None:
            seed = np.random.randint(123456789)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.size = torch.tensor(size, device=device)
        self.num_instances = num_instances
        self.single_instance = single_instance
        self.sym_dist = sym_dist
        self.sym_flow = sym_flow
        self.tsp_ver = tsp_ver

        # COORDS
        if single_instance:
            # generates random x y coordinates for the locations
            self.coords = torch.rand(1, size, 2, device=device)

            # repeat coords for num_instances
            self.coords = self.coords.repeat(num_instances, 1, 1)
        else:
            self.coords = torch.rand(num_instances, size, 2, device=device)

        # DISTANCES
        # populates dists matrix with coords
        self.dists = torch.cdist(self.coords, self.coords)
        if not sym_dist:
            # make distance matrix asymmetric with changing distance from i to j to be manhattan distance
            for i in range(num_instances):
                for j in range(size):
                    for k in range(j, size):
                        if j != k:
                            self.dists[i, j, k] = torch.abs(
                                self.coords[i, j, 0] - self.coords[i, k, 0]
                            ) + torch.abs(self.coords[i, j, 1] - self.coords[i, k, 1])

        # FLOWS
        if not tsp_ver:
            if single_instance:
                # create flow matrix with values from [0, 1)
                self.flows = torch.rand(1, size, size, device=device)

                # repeat flows for num_instances
                self.flows = self.flows.repeat(num_instances, 1, 1)
            else:
                self.flows = torch.rand(num_instances, size, size, device=device)

            self.flows[:, range(size), range(size)] = 0.0

            if sym_flow:
                self.flows = 0.5 * (self.flows + self.flows.transpose(1, 2))

        else:
            self.flows = torch.zeros(1, size, size)

            indices = torch.arange(1, size)
            self.flows[0, indices - 1, indices] = 1
            self.flows[0, size - 1, 0] = 1

            self.flows = self.flows.repeat(num_instances, 1, 1)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, index):
        return {
            "size": self.size,
            "coords": self.coords[index],
            "dists": self.dists[index],
            "flows": self.flows[index],
        }

    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)

    def print_data(self, index):
        """Prints problem_instance data"""

        problem_instance = self.__getitem__(index)
        print("Problem Size:")
        print(problem_instance["size"])
        print("X Y Coordinates:")
        print(problem_instance["coords"])
        print("Distance Matrix:")
        print(problem_instance["dists"])
        print("Flow Matrix:")
        print(problem_instance["flows"])


def gurobi_solve(problem_instance, time_limit_seconds=600):
    model = gp.Model("GurobiSolve")

    np.warnings.filterwarnings("ignore")
    model.setParam("LogToConsole", 0)
    model.setParam("TimeLimit", time_limit_seconds)

    model.params.NonConvex = 2

    # create_perm_matrix -- boolean to determine type of gurobi solve
    # Solving for True is infeasable for sufficiently large problem sizes
    # Create a 2-D array of binary variables
    x = model.addMVar(
        (problem_instance["size"], problem_instance["size"]),
        vtype=GRB.BINARY,
        name="x",
    )

    cost_matrix = model.addMVar(
        (problem_instance["size"], problem_instance["size"]),
    )

    dists = problem_instance["dists"].squeeze(0).numpy()
    flows = problem_instance["flows"].squeeze(0).numpy()

    # F * (X @ D @ X^T)
    model.addConstr(cost_matrix == (flows * (x @ dists @ x.T)))

    # minimize sum of cost matrix
    model.setObjective(cost_matrix.sum(), GRB.MINIMIZE)

    # Add row and column constraints
    for i in range(problem_instance["size"]):
        # Sum of each row is 1
        model.addConstr(x[i, :].sum() == 1, name="row" + str(i))

        # Sum of each column is 1
        model.addConstr(x[:, i].sum() == 1, name="col" + str(i))

    # Optimize model
    model.optimize()

    return {
        "x": torch.tensor(x.X),
        "cost": torch.tensor(cost_matrix.sum().getValue()),
    }


def iden_solve(problem_instance):
    b_size = problem_instance["size"].shape[0]

    cost_matrix = problem_instance["flows"] * problem_instance["dists"]

    return {
        "x": torch.eye(problem_instance["size"][0]).unsqueeze(0).repeat(b_size, 1, 1),
        "cost": torch.sum(cost_matrix, dim=(1, 2)),
    }


def pick_best_swap(permutation, problem_instance):
    """Greedily picks best permutation from neighborhood of 1 row-swap away permutations"""

    # Creates a temp and best permutation
    temp_permutation = torch.clone(permutation)
    best_permutation = torch.clone(permutation)

    i = 0
    j = 1

    while i != j:
        # TODO: figure out if by changing permutation to best_permutation it becomes swap_solve
        # 1, problem_size, problem_size -> problem_size, problem_size
        temp_permutation = torch.clone(permutation).squeeze(0)

        # swap rows i and j
        temp_permutation[[i, j]] = temp_permutation[[j, i]]
        # problem_size, problem_size -> 1, problem_size, problem_size
        temp_permutation = temp_permutation.unsqueeze(0)

        # Evaluate the current permutation vs best permutation
        if reward(problem_instance, temp_permutation) < reward(
            problem_instance, best_permutation
        ):
            best_permutation = torch.clone(temp_permutation)

        if j == problem_instance["size"][0] - 1:
            i += 1
            j = i + 1
            if i == problem_instance["size"][0] - 1:
                break
        else:
            j += 1

    # Return best permutation 1 row-swap away from original permutation
    return best_permutation


def swap_solve(problem_instance):
    """Greedy solution to QAP, implements pick_best_swap until convergence"""
    batch_size = problem_instance["size"].shape[0]
    n = problem_instance["size"][0]

    perms, rewards = [], []

    for i in range(batch_size):
        instance = {k: v[i].unsqueeze(0) for k, v in problem_instance.items()}

        permutation = torch.eye(n).unsqueeze(0)
        permutation_prime = torch.clone(permutation)

        iteration = 0
        # Keeps picking best swap until reached minimum
        while iteration <= 1000:
            prev_permutation = torch.clone(permutation_prime)
            permutation_prime = pick_best_swap(permutation_prime, instance)

            # check if the permutation is the same as the previous permutation
            if torch.equal(permutation_prime, prev_permutation):
                break

            iteration += 1

        perms.append(permutation_prime)

    perms = torch.cat(perms, dim=0)

    return [perms, reward(problem_instance, perms)]


def reward(problem_instance, permutation_matrix):
    # batch_size x size x size
    cost_matrix = problem_instance["flows"] * torch.bmm(
        permutation_matrix,
        torch.bmm(problem_instance["dists"], permutation_matrix.transpose(1, 2)),
    )

    # batch_size
    return torch.sum(cost_matrix, dim=(1, 2))


def intermediate_reward_dual_pointer(problem_instance, ast):
    batch_size = problem_instance["size"].shape[0]
    problem_size = problem_instance["size"][0]

    inter_rewards = []

    for i in range(batch_size):
        flows = problem_instance["flows"][i]
        dists = problem_instance["dists"][i]
        assignment_tuples = ast[i]

        intermediate_rewards = torch.zeros(2 * problem_size, device=flows.device)

        for i in range(problem_size):
            for j in range(0, i):
                intermediate_rewards[2 * i + 1] += (
                    dists[assignment_tuples[i][0], assignment_tuples[j][0]]
                    * flows[assignment_tuples[i][1], assignment_tuples[j][1]]
                ) + (
                    dists[assignment_tuples[j][0], assignment_tuples[i][0]]
                    * flows[assignment_tuples[j][1], assignment_tuples[i][1]]
                )

        inter_rewards.append(intermediate_rewards)

    # batch_size, 2 * problem_size
    return torch.stack(inter_rewards)


def intermediate_reward(problem_instance, prev_picks, pick):
    batch_size = problem_instance["size"].shape[0]
    problem_size = problem_instance["size"][0]

    inter_rewards = []

    if torch.equal(
        prev_picks,
        torch.empty(batch_size, 0, 2, dtype=torch.long, device=prev_picks.device),
    ):
        return torch.zeros(batch_size, device=prev_picks.device)

    for i in range(batch_size):
        flows = problem_instance["flows"][i]
        dists = problem_instance["dists"][i]

        pick_reward = 0

        prev_pick = prev_picks[i]  # num_picks, 2
        curr_pick = pick[i]  # 2

        for j in range(prev_pick.shape[0]):
            pick_reward += (
                dists[prev_pick[j][0], curr_pick[0]]
                * flows[prev_pick[j][1], curr_pick[1]]
            ) + (
                dists[curr_pick[0], prev_pick[j][0]]
                * flows[curr_pick[1], prev_pick[j][1]]
            )

        inter_rewards.append(pick_reward)

    # batch_size
    return torch.stack(inter_rewards).to(prev_picks.device)
