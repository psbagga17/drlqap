import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

sys.path.append("..")

import QAP as QAP
from linear_terms.eval import Eval


class Encoder_Attention(nn.Module):
    def __init__(self, problem_size, intermediate_size, rnn_size, dropout):
        super(Encoder_Attention, self).__init__()

        self.problem_size = problem_size
        self.intermediate_size = intermediate_size
        self.rnn_size = rnn_size

        self.sa = nn.MultiheadAttention(rnn_size, 8, batch_first=True, dropout=dropout)
        self.linear_1 = nn.Linear(rnn_size, rnn_size)

    def forward(self, x) -> torch.Tensor:
        # batch_size, problem_size, rnn_size
        x_hat = x + self.sa(x, x, x)[0]
        x_f = x_hat + self.linear_1(x_hat)

        return x_f


class Encoder_Coords(nn.Module):
    def __init__(self, problem_size, intermediate_size, rnn_size, attn, dropout):
        super(Encoder_Coords, self).__init__()

        self.problem_size = problem_size
        self.conv_coord_1 = nn.Conv1d(2 * problem_size, intermediate_size, 1, 1)
        self.conv_coord_2 = nn.Conv1d(intermediate_size, intermediate_size * 2, 1, 1)
        self.conv_coord_3 = nn.Conv1d(intermediate_size * 2, rnn_size, 1, 1)
        self.conv_coord_4 = nn.Conv1d(rnn_size, rnn_size, 1, 1)

        self.attn = attn
        self.attention = (
            Encoder_Attention(problem_size, intermediate_size, rnn_size, dropout)
            if attn
            else None
        )

    def forward(self, embedding_input) -> torch.Tensor:
        # embedding_input -- batch_size, k, 2*problem_size
        # coords matrix & linear terms

        # batch_size, k, 2*problem_size -> batch_size, 2*problem_size, k
        embedding_input = embedding_input.permute(0, 2, 1)

        embeddings = self.conv_coord_1(embedding_input)
        embeddings = self.conv_coord_2(embeddings)
        embeddings = self.conv_coord_3(embeddings)
        embeddings = self.conv_coord_4(embeddings)

        # batch_size, rnn_size, k -> batch_size, k, rnn_size
        embeddings = embeddings.permute(0, 2, 1)

        if self.attn:
            embeddings = self.attention(embeddings)

        global_embeddings = torch.mean(
            embeddings, dim=1, keepdim=True
        )  # batch_size, 1, rnn_size

        return embeddings, global_embeddings


class Encoder_Flows(nn.Module):
    def __init__(self, problem_size, intermediate_size, rnn_size, attn, dropout):
        super(Encoder_Flows, self).__init__()

        self.problem_size = problem_size
        self.sage1 = SAGEConv(problem_size, intermediate_size, normalize=True)
        self.sage2 = SAGEConv(intermediate_size, 2 * intermediate_size, normalize=True)
        self.sage3 = SAGEConv(2 * intermediate_size, intermediate_size, normalize=True)
        self.sage4 = SAGEConv(intermediate_size, rnn_size, normalize=True)
        self.pool = global_mean_pool

        self.attn = attn
        self.attention = (
            Encoder_Attention(problem_size, intermediate_size, rnn_size, dropout)
            if attn
            else None
        )

    def forward(self, flow_matrix) -> torch.Tensor:
        # flow_matrix -- batch_size, k, problem_size
        batch_size, k, _ = flow_matrix.shape

        # batch_size*k, problem_size
        node_features = flow_matrix.view(-1, self.problem_size)

        edge_index = (
            torch.combinations(torch.arange(k, device=flow_matrix.device), r=2)
            .t()
            .contiguous()
        )
        edge_index = edge_index.repeat(1, batch_size)

        x = self.sage1(node_features, edge_index)
        x = self.sage2(x, edge_index)
        x = self.sage3(x, edge_index)
        x = self.sage4(x, edge_index)
        x = F.relu(x)

        # batch_size, k, rnn_size
        x = x.reshape(batch_size, k, -1)

        return x, 1


class Worker(nn.Module):
    def __init__(self, problem_size, intermediate_size, rnn_size, workers, dropout):
        super(Worker, self).__init__()

        self.problem_size = problem_size
        self.intermediate_size = intermediate_size
        self.rnn_size = rnn_size
        self.num_workers = workers

        self.Q_CAs = nn.ModuleList(
            [nn.Linear(rnn_size, rnn_size, bias=False) for _ in range(workers)]
        )
        self.K_CAs = nn.ModuleList(
            [nn.Linear(rnn_size, rnn_size, bias=False) for _ in range(workers)]
        )

        self.mixing_CA = nn.Linear(workers, 1, bias=False)

        # self.mixing_CA = nn.Sequential(
        #     nn.Linear(2 * problem_size * rnn_size, workers),
        #     nn.LeakyReLU(),
        #     nn.Linear(workers, 1, bias=False),
        # )

        self.Q_SAs = nn.ModuleList(
            [nn.Linear(rnn_size, rnn_size, bias=False) for _ in range(workers)]
        )
        self.K_SAs = nn.ModuleList(
            [nn.Linear(rnn_size, rnn_size, bias=False) for _ in range(workers)]
        )
        self.mixing_SA = nn.Linear(workers, 1, bias=False)

        # self.mixing_SA = nn.Sequential(
        #     nn.Linear(2 * problem_size * rnn_size, workers),
        #     nn.LeakyReLU(),
        #     nn.Linear(workers, 1, bias=False),
        # )

        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b, s_1, s_2) -> torch.Tensor:
        # a, b: batch_size, k, rnn_size
        norm = torch.sqrt(torch.tensor(self.rnn_size).float())

        k = a.size(1)

        # CROSS ATTENTION between flows and coords
        # batch_size, workers, k, rnn_size
        Q_CA = torch.cat([self.dropout(q(a)).unsqueeze(1) for q in self.Q_CAs], dim=1)
        K_CA = torch.cat([self.dropout(k(b)).unsqueeze(1) for k in self.K_CAs], dim=1)
        # batch_size, workers, k, k
        ca = torch.matmul(Q_CA, K_CA.transpose(2, 3)) / norm

        # V1 use mixing function
        # # batch_size, k, k
        # ca = self.mixing_CA(ca.permute(0, 2, 3, 1)).squeeze(-1)
        # V1 use mixing function

        # V2 use each s_1 for time step
        # turn ca into batch_size, k, k*workers
        ca = ca.permute(0, 2, 3, 1).reshape(-1, k, k * self.num_workers)
        ca = s_1(ca)  # batch_size, k, k
        # V2 use each s_1 for time step

        # SELF ATTENTION between output
        Q_SA = torch.cat([self.dropout(q(a)).unsqueeze(1) for q in self.Q_SAs], dim=1)
        K_SA = torch.cat([self.dropout(k(a)).unsqueeze(1) for k in self.K_SAs], dim=1)
        # batch_size, workers, k, k
        sa = torch.matmul(Q_SA, K_SA.transpose(2, 3)) / norm

        # V1 use mixing function
        # # batch_size, k, k
        # sa = self.mixing_SA(sa.permute(0, 2, 3, 1)).squeeze(-1)
        # V1 use mixing function

        # V2 use each s_2 for time step
        # turn sa into batch_size, k, k*workers
        sa = sa.permute(0, 2, 3, 1).reshape(-1, k, k * self.num_workers)
        sa = s_2(sa)  # batch_size, k, k
        # V2 use each s_2 for time step

        return sa


class Model(nn.Module):
    def __init__(
        self,
        problem_size,
        intermediate_size,
        rnn_size,
        workers,
        attn=False,
        dropout=0.0,
    ):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.problem_size = problem_size
        self.intermediate_size = intermediate_size
        self.rnn_size = rnn_size
        self.attn = attn

        self.encoder_coords = Encoder_Coords(
            problem_size, intermediate_size, rnn_size, attn, dropout
        )
        self.encoder_flows = Encoder_Flows(
            problem_size, intermediate_size, rnn_size, attn, dropout
        )

        self.aff = Worker(problem_size, intermediate_size, rnn_size, workers, dropout)

        self.s_1s = nn.ModuleList(
            [
                nn.Linear((problem_size - i) * workers, problem_size - i, bias=False)
                for i in range(problem_size)
            ]
        )
        self.s_2s = nn.ModuleList(
            [
                nn.Linear((problem_size - i) * workers, problem_size - i, bias=False)
                for i in range(problem_size)
            ]
        )

        # xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def action(self, problem_instance, state=None):
        dev = problem_instance["flows"].device
        batch_size = problem_instance["flows"].size(0)

        if state is None:
            state = {}
            state["picks"] = torch.empty(batch_size, 0, 2, dtype=torch.long, device=dev)
            mask_c = torch.ones(batch_size, self.problem_size, device=dev).bool()
            mask_f = torch.ones(batch_size, self.problem_size, device=dev).bool()
            state["mask_c"] = mask_c
            state["mask_f"] = mask_f
            state["t"] = 0

        k = self.problem_size - state["t"]

        flows = self.prepare_flows(
            problem_instance["flows"], state["mask_f"], state["t"]
        )
        coords = self.prepare_coords(
            problem_instance,
            (state["mask_c"], state["mask_f"]),
            state["t"],
            state["picks"],
        )

        flows = self.rms_norm(flows)
        coords = self.rms_norm(coords)

        # batch_size, k, d_k
        f, _ = self.encoder_flows(flows)
        c, _ = self.encoder_coords(coords)

        f = self.rms_norm(f)
        c = self.rms_norm(c)

        # batch_size, k, k
        affinity = self.aff(f, c, self.s_1s[state["t"]], self.s_2s[state["t"]])
        pick, ij, log_prob, entropy, distribution = self.pick(affinity, k)

        action, new_state = self.update(ij, state)
        reward = QAP.intermediate_reward(problem_instance, state["picks"], action)

        return {
            "state": state,
            "action": action,  # actual ij for the perm matrix
            "log_prob": log_prob,
            "entropy": entropy,
            "reward": reward,
            "distribution": distribution,
            "pick": pick,  # index in the distribution
            "new_state": new_state,
        }

    def update(self, pick, state):
        # convert mask from batch, problem
        # to batch, k where k is the index of the True in the mask
        b_size = state["mask_f"].shape[0]

        p_f = torch.nonzero(state["mask_f"], as_tuple=False)[:, 1].reshape(b_size, -1)
        p_c = torch.nonzero(state["mask_c"], as_tuple=False)[:, 1].reshape(b_size, -1)

        s_f = p_f[torch.arange(b_size), pick[0]]
        s_c = p_c[torch.arange(b_size), pick[1]]

        new_mask_f = state["mask_f"].scatter(1, s_f.unsqueeze(1), False)
        new_mask_c = state["mask_c"].scatter(1, s_c.unsqueeze(1), False)

        pick = torch.stack((s_f, s_c), dim=1)
        new_picks = torch.cat((state["picks"], pick.unsqueeze(1)), dim=1)

        new_state = {}
        new_state["picks"] = new_picks
        new_state["mask_f"] = new_mask_f
        new_state["mask_c"] = new_mask_c
        new_state["t"] = state["t"] + 1

        return pick, new_state

    def pick(self, affinity, k) -> torch.Tensor:
        # batch_size, k^2
        rollout = affinity.flatten(start_dim=1)
        rollout = F.softmax(rollout, dim=1)

        distribution = torch.distributions.Categorical(rollout)
        if self.training:
            pick = distribution.sample()
            log_prob = distribution.log_prob(pick)
            entropy = distribution.entropy()
        else:
            pick = rollout.argmax(dim=1)
            log_prob = distribution.log_prob(pick)
            entropy = distribution.entropy()

        i = pick // k
        j = pick % k

        return pick, (i, j), log_prob, entropy, distribution

    def rms_norm(self, x):
        # batch, k, d_k
        return x / torch.sqrt(torch.mean(x**2, dim=2, keepdim=True))

    def prepare_flows(self, flows, mask_f, t) -> torch.Tensor:
        # flows -- batch_size, problem_size, problem_size
        # mask_f -- batch_size, problem_size

        k = self.problem_size - t

        # remove rows and columns, then pad with -1s
        # from flows, we want to delete the rows and columns that are masked
        flows = torch.masked_select(flows, mask_f.unsqueeze(2))
        flows = flows.view(-1, k, self.problem_size)

        flows = torch.masked_select(flows, mask_f.unsqueeze(1))
        flows = flows.view(-1, k, k)

        flows = torch.cat(
            (
                flows,
                -1
                * torch.ones(flows.size(0), flows.size(1), self.problem_size - k).to(
                    flows.device
                ),
            ),
            dim=2,
        )

        # batch_size, k, k
        return flows

    def prepare_coords(self, problem_instance, masks, t, picks) -> torch.Tensor:
        # coords -- batch_size, problem_size, 2
        # mask_c -- batch_size, problem_size

        coords = problem_instance["coords"]
        mask_c, mask_f = masks

        # batch_size, k, 2
        k = self.problem_size - t
        coords = torch.masked_select(coords, mask_c.unsqueeze(2)).view(-1, k, 2)
        # batch_size, k, k
        dists = torch.cdist(coords, coords)

        dists = torch.cat(
            (
                dists,
                -1
                * torch.ones(dists.size(0), dists.size(1), self.problem_size - k).to(
                    dists.device
                ),
            ),
            dim=2,
        )

        # batch_size, problem_size, problem_size
        l_t = self.linear_terms(problem_instance, picks, masks)

        # batch_size, k, k
        l_t = torch.masked_select(l_t, mask_f.unsqueeze(2)).view(
            -1, k, self.problem_size
        )
        l_t = torch.masked_select(l_t, mask_c.unsqueeze(1)).view(-1, k, k)

        # transpose l_t, currently ij is facility i at location j
        # coords row is x_i, y_i, cost of facility k at location i (k in all possible remaining facilities)
        l_t = l_t.permute(0, 2, 1)

        # add on all -1s to reach problem_size
        # batch_size, k, problem_size
        l_t = torch.cat(
            (
                l_t,
                -1
                * torch.ones(l_t.size(0), l_t.size(1), self.problem_size - k).to(
                    l_t.device
                ),
            ),
            dim=2,
        )

        # batch_size, n, problem_size+2
        coords = torch.cat((dists, l_t), dim=2)

        return coords

    def linear_terms(self, problem_instance, picks, masks) -> torch.Tensor:
        # linear cost of picking X_ij (facility i at location j)
        # based off previous picks

        batch_size = problem_instance["coords"].size(0)
        # rewards = torch.zeros(batch_size, self.problem_size, self.problem_size)
        lt = -1 * torch.ones(
            batch_size,
            self.problem_size,
            self.problem_size,
            device=problem_instance["coords"].device,
        )

        if picks.size(1) == 0:
            return lt

        mask_c, mask_f = masks

        possible_picks = torch.nonzero(
            mask_f.unsqueeze(2) & mask_c.unsqueeze(1), as_tuple=False
        )[:, 1:]
        possible_picks_2 = possible_picks.view(batch_size, -1, 2)

        for i in range(possible_picks_2.shape[1]):
            curr_picks = torch.stack(
                (possible_picks_2[:, i, 0], possible_picks_2[:, i, 1]), dim=1
            )

            # batch_size
            ttemp = QAP.intermediate_reward(problem_instance, picks, curr_picks)
            lt[torch.arange(batch_size), curr_picks[:, 0], curr_picks[:, 1]] = ttemp

        return lt

    def make_perm_matrix(self, picks):
        batch_size = picks.size(0)
        perm_matrix = torch.zeros(batch_size, self.problem_size, self.problem_size)
        rows = picks[:, :, 0].reshape(-1)
        cols = picks[:, :, 1].reshape(-1)
        batch_indices = (
            torch.arange(batch_size)
            .view(-1, 1)
            .expand(-1, self.problem_size)
            .reshape(-1)
        )
        perm_matrix[batch_indices, rows, cols] = 1

        return perm_matrix

    def forward(self, problem_instance) -> torch.Tensor:
        l_probs = []
        entropies = []
        rewards = []
        states = []

        for i in range(self.problem_size):
            if i == 0:
                o = self.action(problem_instance)
            else:
                o = self.action(problem_instance, o["new_state"])
            l_probs.append(o["log_prob"])
            entropies.append(o["entropy"])
            rewards.append(o["reward"])
            states.append(o["state"])

        # seq_len + 1
        states.append(o["new_state"])

        return {
            "picks": o["new_state"]["picks"],
            "l_probs": torch.stack(l_probs, dim=1),
            "entropy": torch.stack(entropies, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "states": states,
            "perm_matrix": self.make_perm_matrix(o["new_state"]["picks"]),
        }


class Critic(nn.Module):
    def __init__(self, problem_size, intermediate_size, rnn_size, dropout=0.1):
        super(Critic, self).__init__()

        self.problem_size = problem_size
        self.intermediate_size = intermediate_size
        self.rnn_size = rnn_size

        self.o = nn.Sequential(
            nn.Linear(2 * (problem_size**2 + problem_size), 2 * rnn_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * rnn_size, rnn_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_size, 1),
        )

    def state(self, problem_instance, state):
        mask_c = state["mask_c"]
        mask_f = state["mask_f"]

        flows = problem_instance["flows"]
        dists = problem_instance["dists"]

        f = flows.flatten(start_dim=1)  # batch_size, problem_size^2
        d = dists.flatten(start_dim=1)  # batch_size, problem_size^2
        masks = torch.cat((mask_c, mask_f), dim=1).to(
            torch.float64
        )  # batch_size, 2 * problem_size

        input = torch.cat(
            (f, d, masks), dim=1
        )  # batch_size, 2 * problem_size^2 + 2 * problem_size

        # a = (v(s') - v(s)) -- computed by hand
        # b = critic(s) -- computed by model

        # return a + abs(b)

        return self.o(input).squeeze(-1)  # batch_size


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--size", type=int, default=4)
#     parser.add_argument("--intermediate_size", type=int, default=128)
#     parser.add_argument("--rnn_size", type=int, default=128)
#     parser.add_argument("--workers", type=int, default=16)
#     parser.add_argument("--attn", action="store_true")
#     parser.add_argument("--dropout", type=float, default=0.1)
#     parser.add_argument("--num_instances", type=int, default=100)
#     parser.add_argument("--single", action="store_true")
#     parser.add_argument("--epochs", type=int, default=150)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--lr", type=float, default=0.001)
#     parser.add_argument("--beta", type=float, default=0.1)
#     parser.add_argument("--wandb", action="store_true")
#     parser.add_argument("--save", action="store_true")
#     parser.add_argument("--row", action="store_true")

#     args = parser.parse_args()

#     seed = args.seed
#     size = args.size
#     intermediate_size = args.intermediate_size
#     rnn_size = args.rnn_size
#     workers = args.workers
#     attn = args.attn
#     dropout = args.dropout
#     num_instances = args.num_instances
#     single = args.single
#     epochs = args.epochs
#     b_size = args.batch_size
#     lr = args.lr
#     beta = args.beta
#     wandb_flag = args.wandb
#     save = args.save
#     row = args.row

#     torch.manual_seed(seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     dataset = QAP.QAPDataset(size, num_instances, single, device=device, seed=seed)
#     data = DataLoader(dataset, b_size, shuffle=False)
#     temp = next(iter(data))

#     model = Model(size, intermediate_size, rnn_size, workers, dropout, False)
#     model.to(device)

#     o = model(temp)
#     print(o["perm_matrix"])
