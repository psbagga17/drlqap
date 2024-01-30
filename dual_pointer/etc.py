import sys

import wandb

sys.path.append("..")
from QAP import QAPDataset


def get_datasets(
    problem_size: int,
    num_instances: int,
    test_instances: int,
    single_instance: bool,
    sym_dist: bool,
    sym_flow: bool,
    tsp_ver: bool,
    seed: int,
    device: str,
    match_dataset: bool,
):
    if match_dataset:
        QAP_train_dataset = QAP_test_dataset = QAP_val_dataset = QAPDataset(
            problem_size,
            num_instances,
            single_instance,
            sym_dist,
            sym_flow,
            tsp_ver,
            device,
            seed,
        )
    else:
        QAP_train_dataset = QAPDataset(
            problem_size,
            num_instances,
            single_instance,
            sym_dist,
            sym_flow,
            tsp_ver,
            device,
            seed,
        )
        QAP_test_dataset = QAPDataset(
            problem_size,
            test_instances,
            single_instance,
            sym_dist,
            sym_flow,
            tsp_ver,
            device,
            seed + 1,
        )
        QAP_val_dataset = QAPDataset(
            problem_size,
            test_instances,
            single_instance,
            sym_dist,
            sym_flow,
            tsp_ver,
            device,
            seed + 2,
        )

    return QAP_train_dataset, QAP_test_dataset, QAP_val_dataset


def plotting(
    upper,
    lower,
    perf,
    config,
):
    if not config["wandb_log"]:
        return

    # filters out every jump epochs
    jump = 2

    # num_steps, batch_size -> batch_size, num_epochs / jump
    perf = perf.transpose(1, 0)[:, ::jump]

    # num_epochs, problem_size, 5, problem_size
    # ->
    # batch_size, problem_size, problem_size, num_epochs / jump
    upper_b = upper.permute(2, 1, 3, 0)[:, :, :, ::jump]
    lower_b = lower.permute(2, 1, 3, 0)[:, :, :, ::jump]

    wandb.log(
        {
            "perf": wandb.plot.line_series(
                xs=[i for i in range(int(perf.shape[-1]))],
                ys=perf,
                keys=[i for i in range(perf.shape[0])],
                title="Performance",
            )
        }
    )

    for j in range(upper_b.shape[0]):
        upper = upper_b[j]
        lower = lower_b[j]
        for i in range(upper.shape[1]):
            plot_probs(
                [upper[i], lower[i]],
                upper.shape[1],
                int(upper.shape[-1]),
                i,
                j,
            )


def plot_probs(
    data_source: list,
    model_size: int,
    num_iterations: int,
    idx: int,
    prefix: str = "",
) -> None:
    # data_source should be of size (model_size, 2)
    # each element of data_source should be of size (model_size, num_iterations)

    wandb.log(
        {
            f"probs_upper_{prefix}_{idx}": wandb.plot.line_series(
                xs=[i for i in range(num_iterations)],
                ys=data_source[0],
                keys=[i for i in range(model_size)],
                title=f"Upper_{prefix}_{idx}",
            ),
            f"probs_lower_{prefix}_{idx}": wandb.plot.line_series(
                xs=[i for i in range(num_iterations)],
                ys=data_source[1],
                keys=[i for i in range(model_size)],
                title=f"Lower_{prefix}_{idx}",
            ),
        }
    )
