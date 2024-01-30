import os
import re
import sys
import time

import torch
from etc import get_datasets
from evaluate import evaluate, truth_evaluate
from model import Critic, Model
from train import save_model, train

save = False  # toggle this for small tests on mac
import sys

sys.path.append("..")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    total_size = 0
    for param in model.parameters():
        num_elements = param.numel()
        element_size = param.element_size()
        total_size += num_elements * element_size
    total_size_gb = total_size / (1024**3)  # Convert bytes to gigabytes
    return total_size_gb


def run(config):
    train_dataset, test_dataset, val_dataset = get_datasets(**config["dataset"])
    save_datasets([train_dataset, test_dataset, val_dataset], config) if save else None

    model = Model(**config["model"]).to(config["device"])
    critic = Critic(**config["critic"]).to(config["device"])

    device_ids = list(range(torch.cuda.device_count()))
    print(device_ids)

    model = torch.nn.DataParallel(model, device_ids=device_ids)
    critic = torch.nn.DataParallel(critic, device_ids=device_ids)

    save_model(model, critic, config, "start")

    print("MODEL PARAMETERS:", count_parameters(model))
    print("CRITIC PARAMETERS:", count_parameters(critic))

    print(f"MODEL SIZE: {get_model_size(model):.3f} GB")
    print(f"CRITIC SIZE: {get_model_size(critic):.3f} GB")

    print("\n----- PRE-EVALUATION -----")
    pre_eval_st_time = time.perf_counter()
    pre_eval = evaluate(model, [test_dataset, val_dataset], config)
    pre_eval_et_time = time.perf_counter()
    print("---- TIME:", pre_eval_et_time - pre_eval_st_time)
    print("----- PRE-EVALUATION -----\n")

    print("\n----- TRAINING -----")
    train_st_time = time.perf_counter()
    train(model, critic, train_dataset, val_dataset, config)
    train_et_time = time.perf_counter()
    print("---- TIME:", train_et_time - train_st_time)
    print("----- TRAINING -----\n")

    print("\n----- POST-EVALUATION -----")
    post_eval_st_time = time.perf_counter()
    post_eval = evaluate(model, [test_dataset, val_dataset], config)
    post_eval_et_time = time.perf_counter()
    print("---- TIME:", post_eval_et_time - post_eval_st_time)
    print("----- POST-EVALUATION -----\n")

    print("\n----- ANALYSIS -----")
    analysis_st_time = time.perf_counter()
    analyze(pre_eval, post_eval, [train_dataset, test_dataset, val_dataset], config)
    analysis_et_time = time.perf_counter()
    print("---- TIME:", analysis_et_time - analysis_st_time)
    print("----- ANALYSIS -----\n")

    print("TOTAL TIME:", analysis_et_time - pre_eval_st_time)


def analyze(
    pre_eval,
    post_eval,
    datasets,
    config,
):
    device_ids = list(range(torch.cuda.device_count()))
    cached_m = Model(**config["model"]).to(config["device"])
    if config["train"]["cache"]:
        cached_m.load_state_dict(
            torch.load(f"checkpoints/{config['timestamp']}/model_cached_best.pt")
        )
        last_epoch = -1
        pattern = re.compile(r"model_cached_(\d+)")
        for filename in os.listdir(f"checkpoints/{config['timestamp']}"):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                if epoch > last_epoch:
                    last_epoch = epoch
        print("BEST EPOCH:", last_epoch)
    cached_m = torch.nn.DataParallel(cached_m, device_ids=device_ids)

    save_datasets(datasets, config) if save else None
    datasets = datasets[1:]

    post_cached_eval = evaluate(cached_m, datasets, config)
    truth = truth_evaluate(datasets, config)

    # optim_gap = {
    #     "greedy": None,
    #     "beam": None,
    #     "swap_greedy": None,
    #     "swap_beam": None,
    # }

    optim_gap = {"greedy": None}

    def percent_diff(a, b):
        return (
            torch.tensor(0, device=config["device"])
            if torch.allclose(a, b)
            else ((a - b) / b)
        )

    for key in pre_eval.keys():
        print("EVAL TYPE:", key)
        # each eval is num_datasets, 2, len(dataset)
        optim_gap_scores = [[] for _ in range(len(datasets))]
        for i, dset in enumerate(datasets):
            pre_data = torch.tensor([], device=config["device"])
            post_data = torch.tensor([], device=config["device"])
            post_cached_data = torch.tensor([], device=config["device"])
            for idx in range(len(dset)):
                trth = truth[i][1][idx]

                pre_data = torch.cat(
                    (
                        pre_data,
                        percent_diff(pre_eval[key][i][1][idx], trth).unsqueeze(0),
                    ),
                    0,
                )
                post_data = torch.cat(
                    (
                        post_data,
                        percent_diff(post_eval[key][i][1][idx], trth).unsqueeze(0),
                    ),
                    0,
                )
                post_cached_data = torch.cat(
                    (
                        post_cached_data,
                        percent_diff(post_cached_eval[key][i][1][idx], trth).unsqueeze(
                            0
                        ),
                    ),
                    0,
                )

            optim_gap_scores[i].append(pre_data)
            optim_gap_scores[i].append(post_data)
            optim_gap_scores[i].append(post_cached_data)

        # num_datasets, 3 (pre, post, post_cached), len(dataset)
        optim_gap_scores = torch.stack([torch.stack(x) for x in optim_gap_scores])
        gap_scores_sorted, _ = torch.sort(optim_gap_scores, dim=2)

        optim_gap[key] = optim_gap_scores

        for i in range(len(datasets)):
            print("DATASET", i)
            print("\nPRE")
            print("AVG PERCENT DIFF", torch.mean(optim_gap_scores[i][0]).item() * 100)
            print("STD PERCENT DIFF", torch.std(optim_gap_scores[i][0]).item() * 100)
            print(
                "PERCENT WITHIN 10%",
                (
                    torch.sum((optim_gap_scores[i][0]) < 0.1)
                    / len(optim_gap_scores[i][0])
                ).item()
                * 100,
            )
            print(
                "95% PERCENTILE",
                gap_scores_sorted[i][0][int(len(datasets[i]) * 0.95)].item() * 100,
            )

            print("\nPOST")
            print("AVG PERCENT DIFF", torch.mean(optim_gap_scores[i][1]).item() * 100)
            print("STD PERCENT DIFF", torch.std(optim_gap_scores[i][1]).item() * 100)
            print(
                "PERCENT WITHIN 10%",
                (
                    torch.sum((optim_gap_scores[i][1]) < 0.1)
                    / len(optim_gap_scores[i][1])
                ).item()
                * 100,
            )
            print(
                "95% PERCENTILE",
                gap_scores_sorted[i][1][int(len(datasets[i]) * 0.95)].item() * 100,
            )

            print("\nPOST-CACHED")
            print("AVG PERCENT DIFF", torch.mean(optim_gap_scores[i][2]).item() * 100)
            print("STD PERCENT DIFF", torch.std(optim_gap_scores[i][2]).item() * 100)
            print(
                "PERCENT WITHIN 10%",
                (
                    torch.sum((optim_gap_scores[i][2]) < 0.1)
                    / len(optim_gap_scores[i][2])
                ).item()
                * 100,
            )
            print(
                "95% PERCENTILE",
                gap_scores_sorted[i][2][int(len(datasets[i]) * 0.95)].item() * 100,
            )

            print("\n----------------\n")

            # save optimality gap scores to checkpoint
            if save:
                os.makedirs("checkpoints", exist_ok=True)
                os.makedirs(f"checkpoints/{config['timestamp']}", exist_ok=True)
                os.makedirs(
                    f"checkpoints/{config['timestamp']}/optim_gap_scores", exist_ok=True
                )
                torch.save(
                    optim_gap_scores,
                    f"checkpoints/{config['timestamp']}/optim_gap_scores/optim_gap_scores_{key}.pt",
                )


def save_datasets(datasets, config):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{config['timestamp']}", exist_ok=True)
    os.makedirs(f"checkpoints/{config['timestamp']}/datasets", exist_ok=True)
    for i, dset in enumerate(datasets):
        torch.save(dset, f"checkpoints/{config['timestamp']}/datasets/dataset_{i}.pt")
