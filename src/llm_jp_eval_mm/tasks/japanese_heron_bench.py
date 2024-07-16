from llm_jp_eval_mm.api.registry import register_task
from llm_jp_eval_mm.api.tasks import Task


@register_task("japanese-heron-bench")
class JapaneseHeronBench(Task):
    def __init__(
        self,
        data_dir="turing-motors/Japanese-Heron-Bench",
        cache_dir=None,
        download_mode=None,
        config=None,
    ):
        super.__init__(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode, config=config)
