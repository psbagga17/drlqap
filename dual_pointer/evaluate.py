import sys

import torch

sys.path.append("..")

from torch.utils.data import DataLoader

from QAP import gurobi_solve, reward, swap_solve


def evaluate(model, datasets, config):
    model.eval()

    # eval_data_types = {
    #     "greedy": [],
    #     "beam": [],
    #     "swap_greedy": [],
    #     "swap_beam": [],
    # }

    eval_data_types = {"greedy": []}

    with torch.no_grad():
        eval_data_greedy = []
        # eval_data_beam = []
        for dataset in datasets:
            eval_data_perm_matrix_g = torch.tensor([], device=config["device"])
            eval_data_reward_g = torch.tensor([], device=config["device"])
            data = DataLoader(
                dataset, batch_size=config["model"]["batch_size"], shuffle=False
            )
            for batch in data:
                out = model.forward(batch)

                eval_data_perm_matrix_g = torch.cat(
                    (eval_data_perm_matrix_g, out[-1]), 0
                )
                eval_data_reward_g = torch.cat(
                    (eval_data_reward_g, reward(batch, out[-1])), 0
                )

            eval_data_greedy.append([eval_data_perm_matrix_g, eval_data_reward_g])

        print("Greedy done")
        # for dataset in datasets:
        #     eval_data_perm_matrix_b = torch.tensor([], device=config["device"])
        #     eval_data_reward_b = torch.tensor([], device=config["device"])
        #     data = DataLoader(dataset, batch_size=1, shuffle=False)
        #     for batch in data:
        #         out_b = model.module.forward_beam(
        #             batch, config["evaluate"]["beam_size"]
        #         )

        #         eval_data_perm_matrix_b = torch.cat(
        #             (eval_data_perm_matrix_b, out_b[-1]), 0
        #         )
        #         eval_data_reward_b = torch.cat(
        #             (eval_data_reward_b, reward(batch, out_b[-1])), 0
        #         )

        #     eval_data_beam.append([eval_data_perm_matrix_b, eval_data_reward_b])

        # print("Beam done")
        eval_data_types["greedy"] = eval_data_greedy
        # eval_data_types["beam"] = eval_data_beam
        # eval_data_types["swap_greedy"] = swap_evaluate_rl(
        #     datasets, eval_data_greedy, config
        # )
        # print("Swap greedy done")
        # eval_data_types["swap_beam"] = swap_evaluate_rl(
        #     datasets, eval_data_beam, config
        # )
        # print("Swap beam done")
    # each eval data in dict is num_datasets, 2, len(dataset)
    return eval_data_types


def swap_evaluate_rl(datasets, eval_data, config):
    swap_eval = []

    for i, dataset in enumerate(datasets):
        swap_eval_perm_matrix = torch.tensor([], device=config["device"])
        swap_eval_reward = torch.tensor([], device=config["device"])
        data = DataLoader(dataset, batch_size=1, shuffle=False)
        for j, batch in enumerate(data):
            ssr = swap_solve_rl(batch, eval_data[i][0][j])
            swap_eval_perm_matrix = torch.cat((swap_eval_perm_matrix, ssr[0]), 0)
            swap_eval_reward = torch.cat((swap_eval_reward, ssr[1]), 0)

        swap_eval.append([swap_eval_perm_matrix, swap_eval_reward])

    # num_datasets, 2, len(dataset)
    return swap_eval


def truth_evaluate(datasets, config):
    if config["evaluate"]["eval_type"] == "gurobi":
        return gurobi_evaluate(datasets)
    elif (
        config["evaluate"]["eval_type"] == "iden"
        or config["evaluate"]["eval_type"] == "identity"
    ):
        return iden_evaluate(datasets)
    elif config["evaluate"]["eval_type"] == "swap":
        return swap_evaluate(datasets)
    else:
        raise ValueError("Invalid evaluation type")


def gurobi_evaluate(datasets):
    gurobi_data = []

    for dataset in datasets:
        dset_perm_matrix = torch.tensor([])
        dset_reward = torch.tensor([])
        data = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in data:
            for k, v in batch.items():
                batch[k] = v.cpu()

            truth = gurobi_solve(batch)

            dset_perm_matrix = torch.cat((dset_perm_matrix, truth[0].unsqueeze(0)), 0)
            dset_reward = torch.cat((dset_reward, truth[1].unsqueeze(0)), 0)

        gurobi_data.append([dset_perm_matrix, dset_reward])

    # num_datasets, 2, len(dataset)
    return gurobi_data


def iden_evaluate(datasets):
    iden_data = []

    for dataset in datasets:
        data = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        batch = next(iter(data))
        truth = iden_solve(batch)
        dset_perm_matrix = truth[0]
        dset_reward = truth[1]

        iden_data.append([dset_perm_matrix, dset_reward])

    # num_datasets, 2, len(dataset)
    return iden_data


def swap_evaluate(datasets):
    swap_data = []

    for dataset in datasets:
        dset_perm_matrix = torch.tensor([])
        dset_reward = torch.tensor([])
        data = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in data:
            for k, v in batch.items():
                batch[k] = v.cpu()

            truth = swap_solve(batch)

            dset_perm_matrix = torch.cat((dset_perm_matrix, truth[0]), 0)
            dset_reward = torch.cat((dset_reward, truth[1]), 0)

        swap_data.append([dset_perm_matrix, dset_reward])

    # num_datasets, 2, len(dataset)
    return swap_data
