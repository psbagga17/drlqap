"""Runs the Model"""

import argparse
import datetime
import sys

import torch
import wandb

sys.path.append("..")

from run import run


def main():
    time = datetime.datetime.now()
    print("----- ----- ", time, " ----- -----")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if torch.__version__[0] == "2":
        print(f"Using torch version {torch.__version__}")
        torch.set_default_device(device)

    parser = argparse.ArgumentParser(description="DRL-QAP-vX")

    parser.add_argument("--problem_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=40)

    parser.add_argument("--rnn_size", type=int, default=128)
    parser.add_argument("--encoder_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epsilon", type=float, default=1.00)
    parser.add_argument("--model_seed", type=int, default=42)

    parser.add_argument("--critic_hidden_size_1", type=int, default=512)
    parser.add_argument("--critic_hidden_size_2", type=int, default=1024)

    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--test_instances", type=int, default=1000)
    parser.add_argument("--single_instance", default=False, action="store_true")
    parser.add_argument("--sym_dist", default=True, action="store_false")
    parser.add_argument("--sym_flow", default=True, action="store_false")
    parser.add_argument("--tsp_ver", default=False, action="store_true")
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--match_dataset", default=False, action="store_true")

    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--BETA", type=float, default=0.0)
    parser.add_argument("--GAMMA", type=float, default=1.0)
    parser.add_argument("--cache", default=False, action="store_true")
    parser.add_argument("--save_params", default=False, action="store_true")

    parser.add_argument("--eval_type", default="swap", type=str)
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--wandb_log", default=False, action="store_true")

    args = parser.parse_args()

    print(f"Using device: {device}")
    print(f"Python version: {sys.version}")

    # config dict
    config = dict(
        architecture="vX",
        seed=args.model_seed,
        device=device,
        timestamp=time.strftime("%Y-%m-%d-%H-%M-%S"),
        model=dict(
            problem_size=args.problem_size,
            batch_size=args.batch_size,
            encoder_size=args.encoder_size,
            rnn_size=args.rnn_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epsilon=args.epsilon,
            device=device,
        ),
        critic=dict(
            problem_size=args.problem_size,
            hidden_size_1=args.critic_hidden_size_1,
            hidden_size_2=args.critic_hidden_size_2,
        ),
        dataset=dict(
            problem_size=args.problem_size,
            num_instances=args.num_instances,
            test_instances=args.test_instances,
            single_instance=args.single_instance,
            sym_dist=args.sym_dist,
            sym_flow=args.sym_flow,
            tsp_ver=args.tsp_ver,
            seed=args.dataset_seed,
            device=device,
            match_dataset=args.match_dataset,
        ),
        train=dict(
            train_epochs=args.train_epochs,
            learning_rate=args.learning_rate,
            BETA=args.BETA,  # beta = 0.0005 worked
            GAMMA=args.GAMMA,  # try gamma = 1.0
            device=device,
            wandb_log=args.wandb_log,
            cache=args.cache,
        ),
        evaluate=dict(
            eval_type=args.eval_type,
            beam_size=min(args.beam_size, args.problem_size),
        ),
        wandb_log=args.wandb_log,
    )

    if config["wandb_log"]:
        wandb.init(
            project="iclr-review",
            # track hyperparameters and run metadata
            config=config,
        )

    print(config)
    run(config)

    if config["wandb_log"]:
        if config["train"]["cache"]:
            wandb.save(f"checkpoints/{config['timestamp']}/model_cached_best.pt")
            wandb.save(f"checkpoints/{config['timestamp']}/critic_cached_best.pt")
            wandb.save(f"checkpoints/{config['timestamp']}/model_trained.pt")
            wandb.save(f"checkpoints/{config['timestamp']}/critic_trained.pt")
        wandb.save(
            f"checkpoints/{config['timestamp']}/optim_gap_scores/optim_gap_scores_greedy.pt"
        )

        for i in range(3):
            wandb.save(f"checkpoints/{config['timestamp']}/datasets/dataset_{i}.pt")

        wandb.finish()


if __name__ == "__main__":
    main()
