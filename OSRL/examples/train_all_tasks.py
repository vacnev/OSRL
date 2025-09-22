from easy_runner import EasyRunner

if __name__ == "__main__":

    exp_name = "benchmark"
    runner = EasyRunner(log_name=exp_name)

    task = [
        # bullet safety gym envs
        # "OfflineBallRun-v0",
        # "OfflineCarRun-v0",
        # "OfflineDroneRun-v0",
        # "OfflineAntRun-v0",
        "OfflineAntCircle-v0",
        # "OfflineBallCircle-v0",
        # "OfflineCarCircle-v0",
        # "OfflineDroneCircle-v0",
    ]

    # policy = ["train_bc", "train_bcql", "train_bearl", "train_coptidice", "train_cpq", "train_ccac"]
    policy = ["train_pdt"]
    cost_limit = [10]
    seed = [0]
    project = ["OSRL"]

    # Do not write & to the end of the command, it will be added automatically.
    template = "nohup python3 -m examples.train.{} --task {}  --cost_limit {} --seed {} --device cuda:1 --project {}"

    train_instructions = runner.compose(template, [policy, task, cost_limit, seed, project])
    runner.start(train_instructions, max_parallel=15)