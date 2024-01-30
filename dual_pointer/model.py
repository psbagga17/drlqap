import math
import sys
from queue import PriorityQueue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv

sys.path.append("..")

from QAP import reward

torch.set_default_dtype(torch.float64)


class Encoder_Coords(nn.Module):
    def __init__(self, size, intermediate_size, rnn_size):
        super(Encoder_Coords, self).__init__()

        self.size = size
        self.conv_coord_1 = nn.Conv1d(2, rnn_size, 1, 1)
        self.conv_coord_2 = nn.Conv1d(rnn_size, intermediate_size, 1, 1)
        self.conv_coord_3 = nn.Conv1d(intermediate_size, rnn_size, 1, 1)

    def forward(self, problem_instance) -> torch.Tensor:
        # batch_size, problem_size, 2 -> batch_size, 2, problem_size
        embedding_input_coords = problem_instance["coords"].permute(0, 2, 1)
        # batch_size, 2, problem_size -> batch_size, rnn_size, problem_size
        embeddings_coords = self.conv_coord_1(embedding_input_coords)
        embeddings_coords = self.conv_coord_2(embeddings_coords)
        embeddings_coords = self.conv_coord_3(embeddings_coords)

        # batch_size, problem_size, rnn_size
        embeddings_coords = embeddings_coords.permute(0, 2, 1)
        return embeddings_coords


class Encoder_Flows(nn.Module):
    def __init__(self, size, intermediate_size, rnn_size):
        super(Encoder_Flows, self).__init__()

        self.size = size
        self.conv_flow_1 = GCNConv(self.size, rnn_size)
        self.conv_flow_2 = GCNConv(rnn_size, intermediate_size)
        self.conv_flow_3 = GCNConv(intermediate_size, rnn_size)

    def forward(self, problem_instance) -> torch.Tensor:
        flow_matrix = problem_instance["flows"]
        batch_size = flow_matrix.size(0)

        x_list = []
        for i in range(batch_size):
            flow = flow_matrix[i]
            edge_index = flow.nonzero(as_tuple=False).t()
            edge_weight = flow[edge_index[0], edge_index[1]].float()

            # problem_size, rnn_size
            x = self.conv_flow_1(flow, edge_index, edge_weight)
            x = self.conv_flow_2(x, edge_index, edge_weight)
            x = self.conv_flow_3(x, edge_index, edge_weight)
            x_list.append(x)

        # batch_size, problem_size, rnn_size
        x = torch.stack(x_list)

        return x


class Pointer(nn.Module):
    def __init__(self, problem_size, rnn_size, num_layers=1, dropout=0.0):
        super(Pointer, self).__init__()

        self.problem_size = problem_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            rnn_size, rnn_size, num_layers, dropout=dropout if num_layers > 1 else 0.0
        )

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, gru_input, h_input):
        self.gru.flatten_parameters()
        # 1, batch_size, rnn_size    D * num_layers, batch_size, rnn_size
        gru_output, h_output = self.gru(gru_input, h_input)

        gru_output = self.drop_rnn(gru_output)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            h_output = self.drop_hh(h_output)

        return gru_output, h_output


class Attention(nn.Module):
    def __init__(self, problem_size, rnn_size, intermediate_size, device):
        super(Attention, self).__init__()

        self.problem_size = problem_size
        self.intermediate_size = intermediate_size
        self.rnn_size = rnn_size

        self.pt_1 = nn.Linear(2 * rnn_size, 1, bias=False)
        self.pt_2 = nn.Linear(2 * rnn_size, 1, bias=False)

    def forward(self, pointer_input, h_t):
        # batch_size, problem_size, rnn_size -> batch_size, rnn_size, problem_size
        pointer_input = pointer_input.permute(0, 2, 1)
        # 1, batch_size, rnn_size -> batch_size, rnn_size, 1
        h_t = h_t.permute(1, 2, 0)

        # batch_size, 2*rnn_size, problem_size
        a_t_input_output = torch.cat(
            (pointer_input, h_t.expand_as(pointer_input)), dim=1
        )

        a_t_input_output = a_t_input_output.permute(0, 2, 1)
        a_t = torch.softmax(self.pt_1(a_t_input_output).squeeze(-1), dim=1)

        c_t = torch.bmm(pointer_input, a_t.unsqueeze(2))  # batch_size, rnn_size, 1

        # batch_size, 2*rnn_size, problem_size
        pi_t_input_output = torch.cat(
            (pointer_input, c_t.expand_as(pointer_input)), dim=1
        )

        pi_t_input_output = pi_t_input_output.permute(0, 2, 1)
        pi_t = self.pt_2(pi_t_input_output).squeeze(-1)

        output = pi_t.unsqueeze(0)

        return output


class Hidden_Sharing(nn.Module):
    def __init__(self, rnn_size, num_layers):
        super(Hidden_Sharing, self).__init__()

        self.upper_proj = nn.Linear(num_layers, rnn_size, bias=False)
        self.lower_proj = nn.Linear(num_layers, rnn_size, bias=False)

    def forward(self, upper, lower):
        upper = upper.permute(1, 2, 0)  # batch_size, rnn_size, num_layers
        lower = lower.permute(1, 2, 0)  # batch_size, rnn_size, num_layers

        u = self.upper_proj(upper)  # batch_size, rnn_size, rnn_size
        l = self.lower_proj(lower)  # batch_size, rnn_size, rnn_size

        mixed = u + l  # batch_size, rnn_size, rnn_size
        add_on = torch.sum(mixed, dim=2).unsqueeze(0)  # 1, batch_size, rnn_size

        upper = upper.permute(2, 0, 1) + add_on  # num_layers, batch_size, rnn_size
        lower = lower.permute(2, 0, 1) + add_on  # num_layers, batch_size, rnn_size

        return upper, lower


def apply_mask(input_tensor, mask_tensor, mask_eps, random_policy, idx, training):
    masked_input = torch.softmax((input_tensor + mask_tensor.log()), dim=2)

    if training:
        # some bug with torch.distributions.Categorical sampling on GPU
        distribution = torch.distributions.Categorical(masked_input)

        # Sample from the distribution (1-epsilon)
        indices_1 = distribution.sample()
        indices_e = random_policy[idx].unsqueeze(0)
        indices = torch.where(mask_eps == 1, indices_e, indices_1)

        probs = torch.gather(masked_input, 2, indices.unsqueeze(2)).squeeze(-1)
        log_probs = distribution.log_prob(indices)  # 1, batch_size
        entropy = distribution.entropy()  # 1, batch_size
    else:
        probs, indices = torch.max(masked_input, dim=2)
        log_probs = probs.log()
        entropy = torch.zeros_like(indices)  # 1, batch_size

    mask_tensor = mask_tensor.scatter_(2, indices.unsqueeze(2), 0)

    # 1, batch_size for indices, probs, log_probs
    # 1, batch_size, problem_size for mask_tensor, masked_input
    return indices, probs, log_probs, entropy, mask_tensor, masked_input


class Model(nn.Module):
    def __init__(
        self,
        problem_size,
        batch_size,  # can ignore this
        rnn_size,
        encoder_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        epsilon=0.0,
        device="cpu",
    ):
        super(Model, self).__init__()

        self.problem_size = problem_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.device = device

        self.dropout = dropout
        self.epsilon = epsilon

        self.enc_coords = Encoder_Coords(problem_size, encoder_size, rnn_size)
        self.enc_flows = Encoder_Flows(problem_size, encoder_size, rnn_size)

        self.ptr_upper = Pointer(problem_size, rnn_size, num_layers, dropout)
        self.ptr_lower = Pointer(problem_size, rnn_size, num_layers, dropout)

        self.attention_upper = Attention(problem_size, rnn_size, hidden_size, device)
        self.attention_lower = Attention(problem_size, rnn_size, hidden_size, device)

        # self.hidden_sharing = Hidden_Sharing(rnn_size, num_layers)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)

    def init_inputs(self, batch_size):
        ptr_upper_input = torch.ones(
            1, batch_size, self.rnn_size, device=self.device, requires_grad=False
        )
        ptr_lower_input = torch.ones(
            1, batch_size, self.rnn_size, device=self.device, requires_grad=False
        )
        h_upper_input = torch.zeros(
            self.num_layers,
            batch_size,
            self.rnn_size,
            device=self.device,
            requires_grad=False,
        )
        h_lower_input = torch.zeros(
            self.num_layers,
            batch_size,
            self.rnn_size,
            device=self.device,
            requires_grad=False,
        )

        mask_upper = torch.ones(
            1,
            batch_size,
            self.problem_size,
            requires_grad=False,
            device=self.device,
        )
        mask_lower = torch.ones(
            1,
            batch_size,
            self.problem_size,
            requires_grad=False,
            device=self.device,
        )

        return (
            ptr_upper_input,
            ptr_lower_input,
            h_upper_input,
            h_lower_input,
            mask_upper,
            mask_lower,
        )

    def forward(self, problem_instance):
        embeddings_coords = self.enc_coords(problem_instance)
        embeddings_flows = self.enc_flows(problem_instance)

        batch_size = embeddings_coords.shape[0]

        (
            ptr_upper_input,
            ptr_lower_input,
            h_upper_input,
            h_lower_input,
            mask_upper,
            mask_lower,
        ) = self.init_inputs(batch_size)

        selector = torch.rand((1, batch_size), requires_grad=False, device=self.device)
        mask_eps = (selector <= self.epsilon).long().to(self.device)
        mask_eps = mask_eps.detach()

        a = torch.stack(
            [
                torch.randperm(self.problem_size, device=self.device)
                for _ in range(batch_size)
            ]
        ).unsqueeze(-1)
        b = torch.stack(
            [
                torch.randperm(self.problem_size, device=self.device)
                for _ in range(batch_size)
            ]
        ).unsqueeze(-1)
        random_policy = torch.cat((a, b), dim=-1)
        random_policy = torch.flatten(random_policy, start_dim=1, end_dim=2).T

        actions = []
        probs = []
        log_probs = []

        dist_upper = []
        dist_lower = []

        entropy = []

        for i in range(self.problem_size):
            # UPPER
            # selecting location
            ptr_upper_output, h_upper_input = self.ptr_upper(
                ptr_upper_input, h_upper_input
            )

            ptr_upper_row = self.attention_upper(embeddings_coords, ptr_upper_output)

            (
                indices_upper,
                probs_upper,
                log_probs_upper,
                entropy_upper,
                mask_upper,
                masked_input_upper,
            ) = apply_mask(
                ptr_upper_row, mask_upper, mask_eps, random_policy, 2 * i, self.training
            )
            dist_upper.append(masked_input_upper.detach())
            actions.append(
                torch.stack(
                    (
                        indices_upper.squeeze(0).detach(),
                        log_probs_upper.squeeze(0).detach(),
                    ),
                    dim=1,
                )
            )
            probs.append(probs_upper)
            log_probs.append(log_probs_upper)
            entropy.append(entropy_upper.squeeze(0))
            ptr_lower_input = embeddings_coords[
                torch.arange(batch_size), indices_upper.squeeze(0)
            ].unsqueeze(0)

            # LOWER
            # selecting facility
            ptr_lower_output, h_lower_input = self.ptr_lower(
                ptr_lower_input, h_lower_input
            )

            ptr_lower_row = self.attention_lower(embeddings_flows, ptr_lower_output)

            (
                indices_lower,
                probs_lower,
                log_probs_lower,
                entropy_lower,
                mask_lower,
                masked_input_lower,
            ) = apply_mask(
                ptr_lower_row,
                mask_lower,
                mask_eps,
                random_policy,
                2 * i + 1,
                self.training,
            )
            dist_lower.append(masked_input_lower.detach())
            actions.append(
                torch.stack(
                    (
                        indices_lower.squeeze(0).detach(),
                        log_probs_lower.squeeze(0).detach(),
                    ),
                    dim=1,
                )
            )
            probs.append(probs_lower)
            log_probs.append(log_probs_lower)
            entropy.append(entropy_lower.squeeze(0))
            ptr_upper_input = embeddings_flows[
                torch.arange(batch_size), indices_lower.squeeze(0)
            ].unsqueeze(0)

            # h_upper_input, h_lower_input = self.hidden_sharing(
            #     h_upper_input, h_lower_input
            # )

        actions = torch.stack(actions, dim=1)  # batch_size, 2 * problem_size, 2
        probs = torch.stack(probs, dim=2).squeeze(0)  # batch_size, 2 * problem_size
        log_probs = torch.stack(log_probs, dim=2).squeeze(
            0
        )  # batch_size, 2 * problem_size
        entropy = torch.stack(entropy, dim=1)  # batch_size, 2 * problem_size

        # batch_size, problem_size*2
        assignments = actions[:, :, 0].detach().type(torch.int)
        # batch_size, problem_size, 2
        assignment_tuples = assignments.view(batch_size, self.problem_size, 2)

        range_tensor = torch.arange(batch_size, device=self.device)
        row_indices = assignment_tuples[range_tensor, :, 1]
        col_indices = assignment_tuples[range_tensor, :, 0]
        # batch_size, problem_size, problem_size
        perm_matrix = torch.zeros(
            batch_size,
            self.problem_size,
            self.problem_size,
            requires_grad=False,
            device=self.device,
        )
        perm_matrix[
            range_tensor[:, None].long(), row_indices.long(), col_indices.long()
        ] = 1

        # problem_size, batch_size, problem_size
        # first problem_size is num_steps that upper passes through (2 * problem_size) for total, so problem_size for upper
        # second problem_size is the possible options to choose from -- then softmaxed later on
        upper = torch.stack(dist_upper, dim=1).squeeze(0)
        lower = torch.stack(dist_lower, dim=1).squeeze(0)

        return (
            assignment_tuples,  # batch_size, problem_size, 2
            probs,  # batch_size, 2 * problem_size
            log_probs,  # batch_size, 2 * problem_size
            entropy,  # batch_size, 2 * problem_size
            actions,  # batch_size, 2 * problem_size, 2 (index, log_prob)
            [upper, lower],  # 2, problem_size, batch_size, problem_size
            perm_matrix,  # batch_size, problem_size, problem_size
        )

    @torch.no_grad()
    def forward_beam(self, problem_instance, k):
        embeddings_coords = self.enc_coords(problem_instance)
        embeddings_flows = self.enc_flows(problem_instance)

        batch_size = embeddings_coords.shape[0]

        (
            ptr_upper_input,
            ptr_lower_input,
            h_upper_input,
            h_lower_input,
            mask_upper,
            mask_lower,
        ) = self.init_inputs(batch_size)

        pq = PriorityQueue()

        ptr_upper_output, h_upper_input, ptr_upper_row = self.ptr_upper(
            ptr_upper_input, h_upper_input
        )
        ptr_upper_row = self.attention_upper(embeddings_coords, ptr_upper_output)
        probs_upper = torch.softmax(ptr_upper_row, dim=2)
        p_upper, indices_upper = torch.topk(probs_upper, k, dim=2)

        for i in range(k):
            pq.put(
                (
                    -p_upper[:, :, i].item(),
                    [indices_upper[:, :, i].item()],
                    mask_upper.clone().scatter_(
                        2, indices_upper[:, :, i].unsqueeze(2), 0
                    ),
                    mask_lower,
                )
            )

        for i in range(self.problem_size):
            num_to_select = k if i < self.problem_size - k else self.problem_size - i
            if i != 0:
                pq_t = PriorityQueue()
                while not pq.empty():
                    item = pq.get()
                    ptr_upper_input = embeddings_flows[
                        torch.arange(batch_size), item[1][-1]
                    ].unsqueeze(0)
                    ptr_upper_output, h_upper_input, ptr_upper_row = self.ptr_upper(
                        ptr_upper_input, h_upper_input
                    )
                    ptr_upper_row = self.attention_upper(
                        embeddings_coords, ptr_upper_output
                    )
                    # 1, batch_size, problem_size
                    probs_upper = torch.softmax((ptr_upper_row + item[2].log()), dim=2)
                    # using torch.where to mask out the already selected indices
                    probs_upper = torch.where(item[2] == 0, -1, probs_upper)
                    p_upper, indices_upper = torch.topk(
                        probs_upper, num_to_select, dim=2
                    )

                    for j in range(p_upper.shape[2]):
                        pq_t.put(
                            (
                                10 * item[0] * p_upper[:, :, j].item(),
                                item[1] + [indices_upper[:, :, j].item()],
                                item[2]
                                .clone()
                                .scatter_(2, indices_upper[:, :, j].unsqueeze(2), 0),
                                item[3],
                            )
                        )
                del pq
                pq = PriorityQueue()
                for j in range(k):
                    pq.put(pq_t.get())
                del pq_t

            # lower
            pq_t = PriorityQueue()
            while not pq.empty():
                item = pq.get()
                ptr_lower_input = embeddings_coords[
                    torch.arange(batch_size), item[1][-1]
                ].unsqueeze(0)
                ptr_lower_output, h_lower_input, ptr_lower_row = self.ptr_lower(
                    ptr_lower_input, h_lower_input
                )
                ptr_lower_row = self.attention_lower(embeddings_flows, ptr_lower_output)
                # 1, batch_size, problem_size
                probs_lower = torch.softmax((ptr_lower_row + item[3].log()), dim=2)
                # using torch.where to mask out the already selected indices
                probs_lower = torch.where(item[3] == 0, -1, probs_lower)
                p_lower, indices_lower = torch.topk(probs_lower, num_to_select, dim=2)

                for j in range(p_lower.shape[2]):
                    pq_t.put(
                        (
                            10 * item[0] * p_lower[:, :, j].item(),
                            item[1] + [indices_lower[:, :, j].item()],
                            item[2],
                            item[3]
                            .clone()
                            .scatter_(2, indices_lower[:, :, j].unsqueeze(2), 0),
                        )
                    )
            del pq
            pq = PriorityQueue()
            for j in range(k):
                pq.put(pq_t.get())
            del pq_t

        min_r = math.inf
        min_perm_matrix = None
        while not pq.empty():
            item = pq.get()
            assignment_tuples = torch.tensor(item[1], device=self.device)
            assignment_tuples = assignment_tuples.view(batch_size, self.problem_size, 2)
            range_tensor = torch.arange(batch_size, device=self.device)
            row_indices = assignment_tuples[range_tensor, :, 1]
            col_indices = assignment_tuples[range_tensor, :, 0]
            # batch_size, problem_size, problem_size
            perm_matrix = torch.zeros(
                batch_size,
                self.problem_size,
                self.problem_size,
                requires_grad=False,
                device=self.device,
            )
            perm_matrix[
                range_tensor[:, None].long(), row_indices.long(), col_indices.long()
            ] = 1
            if reward(problem_instance, perm_matrix) < min_r:
                min_r = reward(problem_instance, perm_matrix)
                min_perm_matrix = perm_matrix

        return (min_r, min_perm_matrix)


class Critic(nn.Module):
    def __init__(self, problem_size, hidden_size_1, hidden_size_2):
        super(Critic, self).__init__()

        self.problem_size = problem_size

        self.seq = nn.Sequential(
            nn.Linear(2 * (problem_size**2 + problem_size), hidden_size_1),
            nn.ELU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ELU(),
            nn.Linear(hidden_size_2, 1),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)

    def forward(self, problem_instance, states):
        # batch_size, problem_size, problem_size -> batch_size, problem_size**2
        flows = problem_instance["flows"].flatten(start_dim=1)
        coords = problem_instance["dists"].flatten(start_dim=1)

        # batch_size, problem_size, 2 -> batch_size, 2 * problem_size
        states = states.flatten(start_dim=1)

        mask = torch.zeros(
            2 * self.problem_size, dtype=torch.bool, device=states.device
        )
        num_steps = 2 * self.problem_size

        init = torch.ones_like(states) * -1
        res = [self.seq(torch.cat((flows, coords, init), dim=1)).squeeze(-1)]

        for step in range(num_steps):
            mask[step] = True
            updated_state = torch.where(mask, states, -1)
            x = torch.cat((flows, coords, updated_state), dim=1)
            x = self.seq(x)
            res.append(x.squeeze(-1))

        # batch_size, num_steps + 1
        res = torch.stack(res, dim=1)

        return res
