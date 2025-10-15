from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import json
from torch.utils.data import Dataset
from .koleg_hparams import KoLEGHyperParams, KoLEGMultimodalHyperParams
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch


def apply_koleg_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: KoLEGHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    icl_examples = ['']

    return icl_examples
