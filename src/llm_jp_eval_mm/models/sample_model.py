from llm_jp_eval_mm.api.model import lmms
from llm_jp_eval_mm.api.registry import register_model


@register_model("sample")
class Sample(lmms):
    """
    Sample Model
    """

    def __init__(self, cfg) -> None:
        super().__init__()
