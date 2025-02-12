# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .transformer_decoder import TransformerDecoder, TransformerDecoderBase, Linear
from .transformer_decoder2 import TransformerDecoder2, TransformerDecoderBase2, Linear
from .transformer_encoder import TransformerEncoder, TransformerEncoderBase
from .transformer_encoder2 import TransformerEncoder2, TransformerEncoderBase2
from .transformer_legacy import (
    TransformerModel,
    TransformerS2Model,
    base_architecture,
    tiny_architecture,
    transformer_iwslt_de_en,
    transformer_wmt_en_de,
    transformer_vaswani_wmt_en_de_big,
    transformer_vaswani_wmt_en_fr_big,
    transformer_wmt_en_de_big,
    transformer_wmt_en_de_big_t2t,
)
from .transformer_base import TransformerModelBase, Embedding
from .transformer_base2 import TransformerModelBase2, Embedding


__all__ = [
    "TransformerModelBase",
    "TransformerModelBase2",
    "TransformerConfig",
    "TransformerDecoder",
    "TransformerDecoderBase",
    "TransformerDecoder2",
    "TransformerDecoderBase2",
    "TransformerEncoder",
    "TransformerEncoderBase",
    "TransformerEncoder2",
    "TransformerEncoderBase2",
    "TransformerModel",
    "TransformerS2Model",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    "transformer_iwslt_de_en",
    "transformer_wmt_en_de",
    "transformer_vaswani_wmt_en_de_big",
    "transformer_vaswani_wmt_en_fr_big",
    "transformer_wmt_en_de_big",
    "transformer_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
