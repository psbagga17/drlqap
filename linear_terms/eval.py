import sys

import torch
from torch.utils.data import DataLoader

sys.path.append("..")

import QAP as QAP


class Eval:
    def __init__(self, config, model, datasets):
        self.config = config

        self.model = model

        self.datasets = datasets[1:]  # just test & val datasets

    def update_model(self, model):
        self.model = model

    def evaluate(self):
        """Evaluate model on test dataset"""
        self.model.eval()
        with torch.no_grad():
            eval_data_greedy = []
            for dataset in self.datasets:
                eval_data_perm_matrix_g = torch.tensor([])
                eval_data_reward_g = torch.tensor([])
                data = DataLoader(
                    dataset,
                    batch_size=self.config["train"]["batch_size"],
                    shuffle=False,
                )
                for batch in data:
                    out = self.model.forward(batch)
                    perm_matrix = out["perm_matrix"]

                    eval_data_perm_matrix_g = torch.cat(
                        (eval_data_perm_matrix_g, perm_matrix), 0
                    )
                    eval_data_reward_g = torch.cat(
                        (eval_data_reward_g, QAP.reward(batch, perm_matrix)), 0
                    )

                # num_datasets, 2, len(dataset)
                eval_data_greedy.append([eval_data_perm_matrix_g, eval_data_reward_g])

        eval_data = {
            "greedy": eval_data_greedy,
            # "swap": # implemented later
            # legacy from dual_pointer code
        }

        return eval_data

    def truth_evaluate(self):
        if self.config["evaluate"]["eval_type"] == "gurobi":
            return self.gurobi_evaluate()
        elif self.config["evaluate"]["eval_type"] == "swap":
            return self.swap_evaluate()
        else:
            raise ValueError("Invalid evaluation type")

    def gurobi_evaluate(self):
        with torch.no_grad():
            gurobi_data = []

            for dataset in self.datasets:
                dset_perm_matrix = torch.tensor([])
                dset_reward = torch.tensor([])
                data = DataLoader(dataset, batch_size=1, shuffle=False)
                for batch in data:
                    for k, v in batch.items():
                        batch[k] = v.cpu()

                    truth = QAP.gurobi_solve(batch)

                    dset_perm_matrix = torch.cat(
                        (dset_perm_matrix, truth[0].unsqueeze(0)), 0
                    )
                    dset_reward = torch.cat((dset_reward, truth[1].unsqueeze(0)), 0)

                gurobi_data.append([dset_perm_matrix, dset_reward])

            # num_datasets, 2, len(dataset)
            return gurobi_data

    def swap_evaluate(self):
        with torch.no_grad():
            swap_data = []

            for dataset in self.datasets:
                dset_perm_matrix = torch.tensor([])
                dset_reward = torch.tensor([])
                data = DataLoader(dataset, batch_size=1, shuffle=False)
                for batch in data:
                    for k, v in batch.items():
                        batch[k] = v.cpu()

                    truth = QAP.swap_solve(batch)

                    dset_perm_matrix = torch.cat((dset_perm_matrix, truth[0]), 0)
                    dset_reward = torch.cat((dset_reward, truth[1]), 0)

                swap_data.append([dset_perm_matrix, dset_reward])

            # num_datasets, 2, len(dataset)
            return swap_data
