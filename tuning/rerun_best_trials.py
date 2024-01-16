import argparse
from typing import List, Tuple

import optuna

import hp_search_spaces


def make_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=
        "Re-run the best trials from a previous tuning run.",
        epilog=f"Example usage:\n"
               f"python rerun_best_trials.py tuning_run.json --top-k 3\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=hp_search_spaces.objectives_by_algo.keys(),
        help="The algorithm that has been tuned. "
             "Can usually be deduced from the study name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many of the top configurations to re-run."
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=5,
        help="How many times to re-run each of the top configurations."
    )
    parser.add_argument(
        "journal_log",
        type=str,
        help="The journal file of the previous tuning run."
    )
    return parser


def infer_algo_name(study: optuna.Study) -> Tuple[str, List[str]]:
    """Infer the algo name from the study name.

    Assumes that the study name is of the form "tuning_{algo}_with_{named_configs_}".

    Args:
        study: The optuna study.

    Returns:
        The algo name.
    """
    study_name_parts = study.study_name.split("_")
    assert len(study_name_parts) >= 3
    assert study_name_parts[0] == "tuning"
    assert study_name_parts[2] == "with"
    return study_name_parts[1]


def get_top_k_trials(study: optuna.Study, k: int) -> List[optuna.trial.Trial]:
    finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(finished_trials) == 0:
        raise ValueError("No trials have completed.")
    if len(finished_trials) < k:
        raise ValueError(
            f"Only {len(finished_trials)} trials have completed, but --top-k is {k}."
        )

    top_k_trials = sorted(
        finished_trials,
        key=lambda t: t.value, reverse=True,
    )[:k]
    return top_k_trials


def main():
    parser = make_parser()
    args = parser.parse_args()
    study: optuna.Study = optuna.load_study(
        storage=optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(args.journal_log)
        ),
        # in our case, we have one journal file per study so the study name can be
        # inferred
        study_name=None,
    )

    top_k_trials = get_top_k_trials(study, args.top_k)

    print("Best trials:")
    for trial in top_k_trials:
        print(trial.value, trial.params)
        print()

    algo_name = args.algo if args.algo is not None else infer_algo_name(study)
    sacred_experiment = hp_search_spaces.objectives_by_algo[algo_name].sacred_ex

    for trial in top_k_trials:
        for i in range(args.num_reruns):
            print(f"Rerunning trial {trial.number} for the {i+1}-th time.")
            result = sacred_experiment.run(
                config_updates=trial.user_attrs["config_updates"],
                named_configs=trial.user_attrs["named_configs"],
                options={"--name": study.study_name, "--file_storage": "sacred"},
            )
            if result.status != "COMPLETED":
                raise RuntimeError(
                    f"Trial failed with {result.fail_trace()} and status {result.status}."
                )


if __name__ == '__main__':
    main()