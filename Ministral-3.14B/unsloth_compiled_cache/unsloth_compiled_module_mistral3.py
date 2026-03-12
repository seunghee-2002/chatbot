"""
2026.3.2
2026.3.4
5.3.0
0.24.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
UNSLOTH_COMPILE_LOCATION = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
if UNSLOTH_COMPILE_LOCATION not in sys.path:
    sys.path.insert(0, UNSLOTH_COMPILE_LOCATION)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass

from unsloth_zoo import DEVICE_TYPE_TORCH, DEVICE_COUNT


from unsloth_zoo.loss_utils import (
    fused_linear_cross_entropy,
    unsloth_fused_ce_loss,
)

if UNSLOTH_STUDIO_ENABLED:
    from unsloth_zoo.loss_utils import fast_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 1, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.autotune_at_compile_time': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.common import torch_compile
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.mistral3.modeling_mistral3 import (__name__, torch, nn, ACT2FN, Cache, GenerationMixin, ModelOutput, BaseModelOutputWithPooling, PreTrainedModel, Unpack, TransformersKwargs, can_return_tuple, Mistral3Config, Mistral3CausalLMOutputWithPast, Mistral3Model, Mistral3PreTrainedModel, Mistral3ForConditionalGeneration)

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Mistral3RMSNorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

class Mistral3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Mistral3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return Mistral3RMSNorm_forward(self, hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def Mistral3PatchMerger_forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    image_sizes = [
        (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
    ]

    tokens_per_image = [h * w for h, w in image_sizes]
    d = image_features.shape[-1]

    permuted_tensor = []
    for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
        grid = torch.nn.functional.unfold(
            image_grid, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size
        )
        grid = grid.view(d * self.spatial_merge_size**2, -1).t()
        permuted_tensor.append(grid)

    image_features = torch.cat(permuted_tensor, dim=0)
    image_features = self.merging_layer(image_features)
    return image_features

class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        return Mistral3PatchMerger_forward(self, image_features, image_sizes)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def Mistral3MultiModalProjector_forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
    image_features = self.norm(image_features)
    image_features = self.patch_merger(image_features, image_sizes)
    hidden_states = self.linear_1(image_features)
    hidden_states = self.act(hidden_states)
    hidden_states = self.linear_2(hidden_states)
    return hidden_states

class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.norm = Mistral3RMSNorm(config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.patch_merger = Mistral3PatchMerger(config)
        # We have hidden_size * the number of vision feature layers
        self.num_feature_layers = (
            1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * self.num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
        return Mistral3MultiModalProjector_forward(self, image_features, image_sizes)


@torch.compiler.disable(recursive = False)
@can_return_tuple
def Mistral3ForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    image_sizes: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Mistral3CausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import httpx
    >>> from io import BytesIO
    >>> from transformers import AutoProcessor, Mistral3ForConditionalGeneration

    >>> model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    >>> processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

    >>> prompt = "<s>[INST][IMG]What is the image?[/INST]"
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> with httpx.stream("GET", url) as response:
    ...     image = Image.open(BytesIO(response.read()))

    >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is the image?The image depicts two cats lying on a pink blanket."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        image_sizes=image_sizes,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None)
        if n_items is None: n_items = (kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        if torch_dynamo_eval_frame is not None:
            old_stance = torch_dynamo_eval_frame._stance.stance
        else:
            old_stance = None
        if old_stance is not None and INFERENCE_RUNS == 1:
            # Skip guards and return to eager -> we still need guards!
            torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Removing compiler guards after 1 inference run. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
        elif old_stance == "eager_on_recompile":
            pass
        elif old_stance == "default" and INFERENCE_RUNS > 1:
            # Reset compiler stance
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
            if UNSLOTH_ENABLE_LOGGING:
                logger_compiler.info(
                    f"Unsloth: Reseting guards. "\
                    f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                    f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                )
            INFERENCE_RUNS = 0
        INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if () == () else (),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () != () else 0,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits=logits, labels=labels.to(self.lm_head.weight.device), vocab_size=self.config.text_config.vocab_size, **kwargs)


    return Mistral3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )

class Mistral3ForConditionalGeneration(Mistral3PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        r"^language_model.model": "model.language_model",
        r"^vision_tower": "model.vision_tower",
        r"^multi_modal_projector": "model.multi_modal_projector",
        r"^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Mistral3Config):
        super().__init__(config)
        self.model = Mistral3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            **kwargs,
        )


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Mistral3CausalLMOutputWithPast:
        return Mistral3ForConditionalGeneration_forward(self, input_ids, pixel_values, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, image_sizes, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            # Pixel values are used only in the first iteration if available
            # In subsequent iterations, they are already merged with text and cached
            # NOTE: first iteration doesn't have to be prefill, it can be the first
            # iteration with a question and cached system prompt (continue generate from cache)
            model_inputs["pixel_values"] = pixel_values

        return model_inputs
