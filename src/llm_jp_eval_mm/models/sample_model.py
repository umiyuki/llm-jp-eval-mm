import warnings

from llm_jp_eval_mm.api.model import lmms
from llm_jp_eval_mm.api.registry import register_model

warnings.filterwarnings("ignore")


@register_model("sample")
class Sample(lmms):
    """
    Sample Model
    """

    def __init__(self) -> None:
        super().__init__()
