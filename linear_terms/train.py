import copy
import datetime
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

sys.path.append("..")

import argparse

from model import Critic, Model

import QAP as QAP
from linear_terms.eval import Eval


class Memory:
    def __init__(self, batch_size, seed=42):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.batch_size = batch_size
        self.seed = seed

    def show(self):
        return (
            self.states,
            self.actions,
            self.logprobs,
            self.rewards,
            self.next_states,
            self.dones,
        )

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]

    def populate(self, state, action, logprob, reward, next_state, done):
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def generate_batches(self):
        n_states = len(self.states)

        batch_start = torch.arange(0, n_states, self.batch_size)
        indices = torch.arange(n_states, dtype=torch.int64)
        indices = torch.randperm(n_states)

        batches = [indices[i : i + self.batch_size] for i in batch_start]

        arrs = self.show()

        # split arrs into n_states / batch_size mini-batches
        # get the indices related to each mini batch and use that to get the correct indices into each mini-batch

        # TODO: optimize this part
        mini_batches = [[[arr[j] for arr in arrs] for j in batch] for batch in batches]

        # C mini-batches of timesteps, each with D state, action, etc information
        # C*D = len(states)
        return mini_batches, batches


class Trainer:
    def __init__(self, config, model, critic, datasets):
        self.config = config

        self.model = model
        self.critic = critic

        self.lr = 0.001

        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.datasets = datasets  # train, test, val

        self.train_type = config["train"]["train_type"]

        self.BETA = config["train"]["BETA"]
        self.GAMMA = config["train"]["GAMMA"]
        self.lambda_ = config["train"]["lambda"]
        self.lr = config["train"]["learning_rate"]

        self.best_perf = math.inf
        self.best_model = None

        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.wandb = config["train"]["wandb"]
        self.output_freq = config["train"]["output_freq"]

        if self.wandb:
            wandb.init(project="linear", config=config)
            wandb.watch(self.model)
            wandb.watch(self.critic)

        self.cache = config["train"]["cache"]

        self.eps = config["train"]["eps"]

        self.mem = Memory(config["train"]["time_batch_size"])
        self.mem_repeats = config["train"]["mem_repeats"]

    def train(self):
        self.save_model("start") if self.cache else None
        scheduler = StepLR(self.model_optimizer, step_size=1, gamma=0.99)
        for epoch in range(self.epochs):
            self.train_epoch()
            self.val_epoch(epoch)
            if self.cache:
                self.save_model(epoch)

            # scheduler.step()

        self.save_model("end") if self.cache else None

    def train_epoch(self):
        self.model.train()
        self.critic.train()

        train_data = DataLoader(self.datasets[0], self.batch_size, shuffle=True)

        cla, clc, cle = 0, 0, 0
        cum_loss = 0

        for batch in train_data:
            if self.train_type == "reinforce":
                loss = self.reinforce_train(batch)
            elif self.train_type == "a2c":
                loss = self.a2c_train(batch)
            elif self.train_type == "ppo":
                loss = self.ppo_train(batch)
            elif self.train_type == "supervised":
                loss = self.supervised_train(batch)

            cla += loss[0]
            clc += loss[1]
            cle += loss[2]

            cum_loss += cla + clc - self.BETA * cle

        wandb.log({"actor_loss": cla}) if self.wandb else None
        wandb.log({"critic_loss": clc}) if self.wandb else None
        wandb.log({"entropy_loss": cle}) if self.wandb else None
        wandb.log({"loss": cum_loss}) if self.wandb else None

    def val_epoch(self, epoch, test=False):
        self.model.eval()
        self.critic.eval()

        val_score = 0

        test_data = DataLoader(self.datasets[-1], self.batch_size, shuffle=False)
        for batch in test_data:
            o = self.model.forward(batch)
            val_score += torch.sum(QAP.reward(batch, o["perm_matrix"]))

        if test:
            return val_score

        if epoch % self.output_freq == 0:
            print(f"Epoch {epoch} | Val: {torch.round(val_score, decimals=2)}")

        if self.wandb and epoch >= 0:
            wandb.log({"val": val_score})

        if val_score < self.best_perf:
            self.best_perf = val_score
            self.best_model = copy.deepcopy(self.model)

        return val_score

    def reinforce_train(self, batch):
        info = self.model.forward(batch)

        # reinforce
        l_probs, rewards, entropy = info["l_probs"], info["rewards"], info["entropy"]

        rewards = -rewards  # minimize rewards

        rewards = (rewards - torch.mean(rewards, dim=0, keepdim=True)) / (
            torch.std(rewards, dim=0, keepdim=True) + 1e-10
        )

        g_t = torch.zeros_like(rewards)
        for i in range(rewards.shape[1]):
            g_t[:, i] = torch.sum(rewards[:, i:], dim=1)

        rewards = g_t

        a_loss = torch.sum(l_probs * rewards, dim=1)
        e_loss = torch.sum(entropy, dim=1)

        loss = a_loss - self.BETA * e_loss
        loss = torch.mean(loss)

        self.model_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        loss.backward()
        self.model_optimizer.step()

        return (a_loss.item(), 0, e_loss.item())

    def a2c_train(self, batch):
        info = self.model.forward(batch)

        l_probs, rewards, entropy = info["l_probs"], info["rewards"], info["entropy"]

        rewards = -rewards  # minimize rewards

        rewards = (rewards - torch.mean(rewards, dim=0, keepdim=True)) / (
            torch.std(rewards, dim=0, keepdim=True) + 1e-10
        )

        vs = []
        for st in info["states"]:
            vs.append(self.critic.state(batch, st))
        # batch_size, seq_len + 1
        vs = torch.stack(vs, dim=1)
        vs[:, -1] = 0

        # batch_size, seq_len
        adv = rewards + (self.GAMMA * vs[:, 1:] - vs[:, :-1])

        a_loss = -torch.sum(l_probs * adv.detach(), dim=1)
        e_loss = torch.sum(entropy, dim=1)
        c_loss = adv.pow(2).mean(1)

        loss = a_loss + c_loss - self.BETA * e_loss

        loss = torch.mean(loss)

        self.model_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        loss.backward()
        self.model_optimizer.step()
        self.critic_optimizer.step()

        return (a_loss.item(), c_loss.item(), e_loss.item())

    def supervised_train(self, batch):
        n = batch["size"][0]

        s_states, s_targets = self.prepare_supervised_data(batch)

        a_loss = 0
        e_loss = 0

        for i in range(n):
            output = self.model.action(batch, s_states[i])

            model_distribution = output["distribution"]
            target_distribution = s_targets[i]

            # loss is mse between the two distributions
            loss = nn.MSELoss()(model_distribution.probs, target_distribution)
            a_loss += loss

            e_loss += model_distribution.entropy().mean()

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

        return (a_loss.item(), 0, e_loss.item())

    def prepare_supervised_data(self, batch):
        swap_perm, swap_r = QAP.swap_solve(batch)

        b_size = batch["size"].shape[0]
        n = batch["size"][0]

        pairs = torch.nonzero(swap_perm, as_tuple=False)[:, 1:].reshape(b_size, -1, 2)

        states = []
        for i in range(n):
            if i == 0:
                states.append(None)
            else:
                state = {}
                state["t"] = i
                selected_so_far = pairs[:, :i]

                state["picks"] = selected_so_far
                mask = torch.ones(b_size, n).bool()

                state["mask_f"] = mask.scatter(1, selected_so_far[:, :, 0], False)
                state["mask_c"] = mask.scatter(1, selected_so_far[:, :, 1], False)

                states.append(state)

        # didnt combine under one loop to make it easier to read
        # still O(n)
        targets = []
        for i in range(n):
            k = n - i

            target = torch.zeros(b_size, n, n)
            selection = pairs[:, i]

            # idx = selection[:, 0] * n + selection[:, 1]
            target[torch.arange(b_size), selection[:, 0], selection[:, 1]] = 1

            if i != 0:
                # now we delete the already selected rows and columns
                target = torch.masked_select(
                    target, states[i]["mask_f"].unsqueeze(2)
                ).view(b_size, k, n)
                target = torch.masked_select(
                    target, states[i]["mask_c"].unsqueeze(1)
                ).view(b_size, k, k)

            targets.append(target.flatten(start_dim=1))

        return states, targets

    def ppo_train(self, batch):
        self.remember(batch)
        batch_loss = self.ppo_update(batch)
        self.mem.clear_memory()
        return batch_loss

    def ppo_update(self, batch):
        mini_batches, batches = self.mem.generate_batches()
        mini_epochs = 3
        tl_a, tl_c, tl_e = 0, 0, 0

        for _ in range(mini_epochs):
            for mini_batch, ts in zip(mini_batches, batches):
                # recompute gaes for every update
                gaes = self.compute_gaes(batch)
                a_loss = 0
                c_loss = 0
                e_loss = 0
                for instance, tss in zip(mini_batch, ts):
                    (
                        state,
                        action,
                        old_logprobs,
                        reward,
                        next_state,
                        done,
                    ) = instance

                    v_s = self.critic.state(batch, state)
                    advantage = gaes[:, tss]

                    pi_new = self.model.action(batch, state)
                    distribution = pi_new["distribution"]

                    new_logprobs = distribution.log_prob(action)
                    entropy = distribution.entropy().mean()

                    ratio = torch.exp(new_logprobs - old_logprobs)

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

                    a_loss += -torch.min(surr1, surr2).mean()
                    c_loss += nn.MSELoss()(v_s, advantage)
                    e_loss += entropy

                self.model_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                loss = a_loss + c_loss - self.BETA * e_loss
                tl_a += a_loss.item()
                tl_c += c_loss.item()
                tl_e += e_loss.item()
                loss.backward()
                self.model_optimizer.step()
                self.critic_optimizer.step()

        return (tl_a, tl_c, tl_e)

    def compute_gaes(self, batch):
        n = batch["size"][0]
        info = self.model.forward(batch)
        # seq_len is equal to problem size / n / etc

        # states = seq len + 1. each state contains info with batch_size
        # rewards batch_size, seq_len
        states, rewards = info["states"], info["rewards"]
        rewards = -rewards  # minimize rewards

        # batch_size, seq_len + 1
        vs = torch.stack([self.critic.state(batch, st) for st in states], dim=1)
        vs[:, -1] = 0

        # batch_size, seq_len
        tds = rewards + (self.GAMMA * vs[:, 1:] - vs[:, :-1])

        gaes = torch.zeros_like(tds)
        for i in reversed(range(n)):
            if i == n - 1:
                gaes[:, i] = tds[:, i]
            else:
                gaes[:, i] = tds[:, i] + self.lambda_ * self.GAMMA * gaes[:, i + 1]

        # gaes = (gaes - torch.mean(gaes, dim=0, keepdim=True)) / (
        #     torch.std(gaes, dim=0, keepdim=True) + 1e-10
        # )

        return gaes

    def remember(self, batch):
        # st, at -> rt, st+1

        for _ in range(self.mem_repeats):
            # for _ in range(1):
            o = self.model.action(batch)
            self.mem.populate(
                o["state"],
                o["pick"],
                o["log_prob"].detach(),
                o["reward"],
                o["new_state"],
                False,
            )
            while True:
                o = self.model.action(batch, o["new_state"])

                flag = False
                if o["new_state"]["t"] == batch["size"][0]:
                    flag = True

                self.mem.populate(
                    o["state"],
                    o["pick"],
                    o["log_prob"].detach(),
                    o["reward"],
                    o["new_state"],
                    flag,
                )

                if flag:
                    break

    def save_model(self, suffix):
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs(f"checkpoints/{self.config['timestamp']}", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"checkpoints/{self.config['timestamp']}/model_{suffix}.pt",
        )
        torch.save(
            self.critic.state_dict(),
            f"checkpoints/{self.config['timestamp']}/critic_{suffix}.pt",
        )


def load_datasets(config, overfit=False):
    if overfit:
        dataset = QAP.QAPDataset(
            config["dataset"]["size"],
            config["dataset"]["num_instances"],
            config["dataset"]["single"],
            config["dataset"]["seed"],
        )

        return [dataset, dataset, dataset]
    else:
        train_dataset = QAP.QAPDataset(
            config["dataset"]["size"],
            config["dataset"]["num_instances"],
            config["dataset"]["single"],
            config["dataset"]["seed"],
        )
        test_dataset = QAP.QAPDataset(
            config["dataset"]["size"],
            config["dataset"]["num_instances"],
            config["dataset"]["single"],
            config["dataset"]["seed"] + 1,
        )
        val_dataset = QAP.QAPDataset(
            config["dataset"]["size"],
            config["dataset"]["num_instances"],
            config["dataset"]["single"],
            config["dataset"]["seed"] + 2,
        )

        return [train_dataset, test_dataset, val_dataset]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=128)
    parser.add_argument("--rnn_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--attn", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--single", action="store_true")
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["reinforce", "a2c", "ppo", "supervised"],
        default="ppo",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tbs", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_", type=float, default=0.75)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--o_freq", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    config = {
        "problem_size": args.size,
        "seed": args.seed,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model": {
            "problem_size": args.size,
            "intermediate_size": args.intermediate_size,
            "rnn_size": args.rnn_size,
            "attn": args.attn,
            "workers": args.workers,
            "dropout": args.dropout,
        },
        "critic": {
            "problem_size": args.size,
            "intermediate_size": args.intermediate_size,
            "rnn_size": args.rnn_size,
            "dropout": args.dropout,
        },
        "evaluate": {
            "eval_type": "swap",
            "beam_size": 1,
        },
        "dataset": {
            "size": args.size,
            "num_instances": args.num_instances,
            "single": args.single,
            "seed": args.seed,
        },
        "train": {
            "train_type": "ppo",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "time_batch_size": args.tbs,
            "mem_repeats": 1,
            "learning_rate": args.lr,
            "eps": args.eps,
            "BETA": args.beta,
            "GAMMA": args.gamma,
            "lambda": args.lambda_,
            "cache": args.cache,
            "output_freq": args.o_freq,
            "wandb": args.wandb,
        },
    }

    datasets = load_datasets(config, overfit=True)

    data = DataLoader(datasets[0], args.batch_size, shuffle=True)
    m = Model(**config["model"])
    c = Critic(**config["critic"])
    t = Trainer(config, m, c, datasets)
    e = Eval(config, m, datasets[1:])

    truth = e.truth_evaluate()
    pre_eval = e.evaluate()["greedy"]

    t.train()

    e.update_model(t.best_model)
    post_eval = e.evaluate()["greedy"]

    # test_dataset
    print("-" * 20)
    print("Mean reward")
    print(f"Truth: {torch.mean(truth[0][1])}")
    print(f"Pre: {torch.mean(pre_eval[0][1])}")
    print(f"Post: {torch.mean(post_eval[0][1])}")
    print("-" * 20)

    print("-" * 20)
    print("Mean % difference")
    print(f"Pre: {100 * torch.mean((pre_eval[0][1] - truth[0][1]) / truth[0][1])}")
    print(f"Post: {100 * torch.mean((post_eval[0][1] - truth[0][1]) / truth[0][1])}")
    print("-" * 20)
