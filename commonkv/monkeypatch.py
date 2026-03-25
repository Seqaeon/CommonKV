from importlib.metadata import version
import transformers

# from commonkv.modeling_llama_svd import LlamaForCausalLM
from commonkv.modeling_llama_svd_merge import LlamaForCausalLM
from commonkv.modeling_mistral_svd_merge import MistralForCausalLM



from commonkv.llama_model import (
    prepare_inputs_for_generation_llama_new,
    llama_attn_forward_PyramidKV,
    llama_attn_forward_L2Norm,
    llama_attn_forward_CAM,
    llama_attn_forward_H2O,
    llama_attn_forward_StreamingLLM,
    llama_attn_forward_SnapKV,
    llama_attn_forward_ThinK,
    llama_attn_forward_MiniCache,
    llama_attn_forward_Palu,
    llama_attn_forward_Custom,
)
from commonkv.mistral_model import (
    prepare_inputs_for_generation_mistral_new,
    mistral_attn_forward_PyramidKV,
    mistral_attn_forward_L2Norm,
    mistral_attn_forward_CAM,
    mistral_attn_forward_H2O,
    mistral_attn_forward_StreamingLLM,
    mistral_attn_forward_SnapKV,
    mistral_attn_forward_ThinK,
    mistral_attn_forward_MiniCache,
    mistral_attn_forward_Palu,
    mistral_attn_forward_Custom,
)


def _replace_llama_attention_forward(method):
    method = method.lower()
    attn_method_map = {
        "pyramidkv": llama_attn_forward_PyramidKV,
        "l2norm": llama_attn_forward_L2Norm,
        "cam": llama_attn_forward_CAM,
        "h2o": llama_attn_forward_H2O,
        "streamingllm": llama_attn_forward_StreamingLLM,
        "snapkv": llama_attn_forward_SnapKV,
        "think": llama_attn_forward_ThinK,
        "palu": llama_attn_forward_Palu,
        "minicache": llama_attn_forward_MiniCache,
        "custom": llama_attn_forward_Custom,
        "apkvc": llama_attn_forward_Custom,
    }
    forward_fn = attn_method_map.get(method)
    if forward_fn is None:
        return
    
    # Patch original Transformers classes
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_fn
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward_fn
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = forward_fn

    # Patch SVD-merged model classes (if commonkv/ours was used)
    try:
        from commonkv import modeling_llama_svd_merge
        modeling_llama_svd_merge.LlamaAttention.forward = forward_fn
        modeling_llama_svd_merge.LlamaFlashAttention2.forward = forward_fn
        modeling_llama_svd_merge.LlamaSdpaAttention.forward = forward_fn
    except (ImportError, AttributeError):
        pass



def _replace_mistral_attention_forward(method):
    method = method.lower()
    attn_method_map = {
        "pyramidkv": mistral_attn_forward_PyramidKV,
        "l2norm": mistral_attn_forward_L2Norm,
        "cam": mistral_attn_forward_CAM,
        "h2o": mistral_attn_forward_H2O,
        "streamingllm": mistral_attn_forward_StreamingLLM,
        "snapkv": mistral_attn_forward_SnapKV,
        "think": mistral_attn_forward_ThinK,
        "palu": mistral_attn_forward_Palu,
        "minicache": mistral_attn_forward_MiniCache,
        "custom": mistral_attn_forward_Custom,
        "apkvc": mistral_attn_forward_Custom,
    }
    forward_fn = attn_method_map.get(method)
    if forward_fn is None:
        return

    # Patch original Transformers classes
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = forward_fn
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = forward_fn
    transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = forward_fn

    # Patch SVD-merged model classes (if commonkv/ours was used)
    try:
        from commonkv import modeling_mistral_svd_merge
        modeling_mistral_svd_merge.MistralAttention.forward = forward_fn
        modeling_mistral_svd_merge.MistralFlashAttention2.forward = forward_fn
        modeling_mistral_svd_merge.MistralSdpaAttention.forward = forward_fn
    except (ImportError, AttributeError):
        pass



def replace_llama(method, model_name=None):

    if method in ["commonkv", "ours"]:
        print("Using CommonKV!")
        transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new
        _replace_llama_attention_forward(method)

    


def replace_mistral(method):

    if method in ["commonkv", "ours"]:
        print("Using CommonKV!")
        transformers.models.mistral.modeling_mistral.MistralForCausalLM = MistralForCausalLM

    
    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new
        _replace_mistral_attention_forward(method)
