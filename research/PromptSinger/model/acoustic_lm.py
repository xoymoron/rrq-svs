# Native Torch architecture lives in acoustic_core; preserve Fairseq registration.
from dataclasses import dataclass
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.utils import safe_getattr
from .acoustic_core import *
from .acoustic_core import AudioboxMegabyteConfig as CoreConfig
from .acoustic_core import AcousticLMT5Configured as CoreModel

@dataclass
class AudioboxMegabyteConfig(CoreConfig, FairseqDataclass):
    pass

@register_model("acoustic_lm", dataclass=AudioboxMegabyteConfig)
class AcousticLMT5Configured(CoreModel, BaseFairseqModel):
    pass

@register_model_architecture(model_name="acoustic_lm", arch_name="acoustic_lm_tiny_nonprefix")
def base_architecture_noprefix(args):
    args.n_state = safe_getattr(args, "n_state", 576)
    args.n_head = safe_getattr(args, "n_head", 8)
    args.n_layer = safe_getattr(args, "n_layer", 12)
    args.prefix_lm = safe_getattr(args, "prefix_lm", False)
    args.dropout = safe_getattr(args, "dropout", 0.0)


@register_model_architecture(model_name="acoustic_lm", arch_name="acoustic_lm_global300M_noprefix")
def architecture_global300M_noprefix(args):
    args.n_state = safe_getattr(args, "n_state", 1152)
    args.n_head = safe_getattr(args, "n_head", 16)
    args.n_layer = safe_getattr(args, "n_layer", 20)
    args.prefix_lm = safe_getattr(args, "prefix_lm", False)
    args.dropout = safe_getattr(args, "dropout", 0.0)

