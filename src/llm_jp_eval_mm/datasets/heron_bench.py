import datasets as hf_datasets


class HeronBench:
    def __init__(self) -> None:
        self.dataset = hf_datasets.load_dataset("turing-motors/Japanese-Heron-Bench")
