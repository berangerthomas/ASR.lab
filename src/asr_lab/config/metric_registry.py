from ..metrics.wer import WER
from ..metrics.cer import CER
from ..metrics.mer import MER
from ..metrics.wil import WIL
from ..metrics.wip import WIP

METRIC_REGISTRY = {
    "wer": WER,
    "cer": CER,
    "mer": MER,
    "wil": WIL,
    "wip": WIP,
}
