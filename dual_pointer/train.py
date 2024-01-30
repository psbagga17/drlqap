import math
import os
import sys

import torch
import torch.nn as nn
import wandb
from etc import plotting
from torch import optim
from torch.utils.data import DataLoader

sys.path.append("..")

import QAP as QAP


def train(
    actor: nn.Module,
    critic: nn.Module,
    train_dataset: DataLoader,
    val_dataset: DataLoader,
    config: dict,
):
    actor_optim = optim.Adam(actor.parameters(), lr=config["train"]["learning_rate"])
    critic_optim = optim.Adam(critic.parameters(), lr=config["train"]["learning_rate"])

    max_grad_norm = 1
    best_performance = math.inf

    batch_size = config["model"]["batch_size"]
    GAMMA = config["train"]["GAMMA"]
    BETA = config["train"]["BETA"]
    wandb_log = config["train"]["wandb_log"]

    if wandb_log:
        wandb.watch([actor, critic], log="gradients", log_freq=1, log_graph=True)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    upper = torch.tensor([]).to(config["device"])
    lower = torch.tensor([]).to(config["device"])
    perf = torch.tensor([]).to(config["device"])

    best_asdf = math.inf

    for epoch in range(config["train"]["train_epochs"]):
        actor.train()
        critic.train()

        info = [0 for _ in range(6)]

        for i, batch in enumerate(train_data):
            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}", end="\n", flush=True)

            (
                assignment_tuples,
                probs,
                log_probs,
                entropies,
                actions,
                distributions,
                perm_matrix,
            ) = actor.forward(batch)

            # batch_size, 2 * problem_size + 1
            est = critic.forward(batch, assignment_tuples)
            # batch_size, 2 * problem_size -- doing gamma * V(s') - V(s)
            est = GAMMA * est[:, 1:] - est[:, :-1]

            inter_rewards = QAP.intermediate_reward_dual_pointer(
                batch, assignment_tuples
            )

            advantage = inter_rewards + est  # batch_size, 2 * problem_size
            policy_grad = -torch.sum(
                torch.mul(log_probs, advantage), dim=1
            )  # batch_size

            # all batch_size
            actor_loss = policy_grad
            critic_loss = torch.mean(torch.pow(advantage, 2), dim=1)
            entropy = torch.sum(entropies, dim=1)

            total_loss = actor_loss - BETA * entropy  # batch_size
            total_loss = torch.mean(total_loss)  # scalar, mean over batch

            actor.eval()
            temp = actor.forward(batch)
            inter_rewards2 = QAP.intermediate_reward_dual_pointer(batch, temp[0])

            best_asdf = min(
                best_asdf, torch.mean(torch.sum(inter_rewards2, dim=1)).item()
            )
            actor.train()

            info[0] += total_loss.item()
            info[1] += torch.mean(actor_loss).item()
            info[2] += torch.mean(critic_loss).item()
            info[3] += torch.mean(entropy).item()
            info[4] += torch.mean(torch.sum(inter_rewards2, dim=1)).item()
            info[5] += torch.mean(torch.sum(est, dim=1)).item()

            # Backprop & update params
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            actor_optim.step()
            critic_optim.step()

        # take the mean over the batches
        info = [i / (len(train_data)) for i in info]

        best_performance = aux_work(
            epoch,
            info[0],
            info[1],
            info[2],
            info[3],
            info[4],
            info[5],
            actor,
            critic,
            val_dataset,
            best_performance,
            config,
        )

        # upper, lower : num_epochs, problem_size, 5, problem_size
        # perf : num_epochs, 5
        upper, lower, perf = logging(actor, train_dataset, upper, lower, perf, config)

    print(best_asdf)

    if wandb_log:
        wandb.unwatch([actor, critic])
    save_model(actor, critic, config, "trained")
    plotting(upper, lower, perf, config)


def aux_work(
    epoch: int,
    total_loss: torch.Tensor,
    actor_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    entropy: torch.Tensor,
    reward: torch.Tensor,
    critic_estimate: torch.Tensor,
    model: nn.Module,
    critic: nn.Module,
    val_dataset: torch.utils.data.Dataset,
    best_perf: float,
    config: dict,
):
    if config["train"]["wandb_log"]:
        wandb.log(
            {
                "loss": total_loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
                "actor_reward": reward,
                "critic_estimate": critic_estimate,
            }
        )

    if epoch % 1 == 0:
        print("-----------------------------------", end="\n", flush=True)
        print("Epoch: ", epoch, end="\n", flush=True)
        print(f"Total Loss: {total_loss}", end="\n", flush=True)
        print(f"Actor Loss: {actor_loss}", end="\n", flush=True)
        print(f"Critic Loss: {critic_loss}", end="\n", flush=True)
        print(
            f"Model Reward: {reward}",
            end="\n",
            flush=True,
        )
        print(f"Critic Output: {critic_estimate}", end="\n", flush=True)
        print(f"Entropy: {entropy}", end="\n", flush=True)
        print("-----------------------------------", end="\n\n", flush=True)

    save_model(model, critic, config, f"cached_{epoch}")

    if config["train"]["cache"]:
        val_data = DataLoader(
            val_dataset, batch_size=config["model"]["batch_size"], shuffle=False
        )
        model.eval()
        critic.eval()
        with torch.no_grad():
            score = 0
            for batch in val_data:
                rewards = QAP.reward(batch, model.forward(batch)[-1])
                score += torch.sum(rewards)
            if score < best_perf:
                best_perf = score
                save_model(model, critic, config, "cached_best")
                print("Best Epoch")
        model.train()
        critic.train()

    return best_perf


def save_model(model, critic, config, suffix):
    if config["train"]["cache"]:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs(f"checkpoints/{config['timestamp']}", exist_ok=True)
        torch.save(
            model.module.state_dict(),
            f"checkpoints/{config['timestamp']}/model_{suffix}.pt",
        )
        torch.save(
            critic.module.state_dict(),
            f"checkpoints/{config['timestamp']}/critic_{suffix}.pt",
        )


def logging(
    model,
    dataset,
    upper,
    lower,
    perf,
    config,
):
    model.eval()

    with torch.no_grad():
        data = DataLoader(
            dataset, batch_size=config["model"]["batch_size"], shuffle=False
        )
        inpt = next(iter(data))
        out = model.forward(inpt)

        upper = torch.cat((upper, out[5][0][:, :5, :].unsqueeze(0)), dim=0)
        lower = torch.cat((lower, out[5][1][:, :5, :].unsqueeze(0)), dim=0)
        perf = torch.cat((perf, QAP.reward(inpt, out[-1])[:5].unsqueeze(0)), dim=0)

    return upper, lower, perf
