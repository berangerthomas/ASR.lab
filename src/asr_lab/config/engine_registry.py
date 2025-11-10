from ..engines.whisper import WhisperEngine
from ..engines.nemo import NeMoEngine
from ..engines.wav2vec2 import Wav2Vec2Engine
from ..engines.vosk import VoskEngine
from ..engines.hubert import HubertEngine
from ..engines.kaldi import KaldiEngine
from ..engines.seamless import SeamlessM4TEngine

ENGINE_REGISTRY = {
    "whisper": WhisperEngine,
    "nemo": NeMoEngine,
    "wav2vec2": Wav2Vec2Engine,
    "vosk": VoskEngine,
    "hubert": HubertEngine,
    "kaldi": KaldiEngine,
    "seamless": SeamlessM4TEngine,
}
