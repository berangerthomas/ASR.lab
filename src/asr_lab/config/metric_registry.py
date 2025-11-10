from ..metrics.wer import WER

METRIC_REGISTRY = {
    "wer": WER,
    # Other metrics like cer, mer, etc., will be added here.
}
