import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer."""
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def get_hidden_states(model, tokens, layer_idx):
    """Extract hidden states from specified layer."""
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    return outputs.hidden_states[layer_idx]

def find_token_position(tokens, target_word, tokenizer):
    """Find position of target word in tokenized sequence."""
    token_strings = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    for i in range(len(token_strings) - 1, -1, -1):
        if target_word in token_strings[i]:
            return i
    return -1

class ComponentPatcher:
    """Context manager for patching component outputs."""
    def __init__(self, model, layer_idx, component_type, patch_tensor, target_position):
        self.model = model
        self.layer_idx = layer_idx
        self.component_type = component_type.lower()
        self.patch_tensor = patch_tensor
        self.target_position = target_position
        self.hook = None

    def _get_target_module(self):
        block = self.model.transformer.h[self.layer_idx]
        if self.component_type == 'mlp':
            return block.mlp
        elif self.component_type == 'attention':
            return block.attn
        else:
            raise ValueError(f"Unknown component_type: {self.component_type}")

    def _patch_hook(self, module, input, output):
        is_attn = isinstance(output, tuple)
        hidden_states = output[0] if is_attn else output
        patched_hidden_states = hidden_states.clone()
        patched_hidden_states[0, self.target_position, :] = self.patch_tensor
        
        if is_attn:
            return (patched_hidden_states,) + output[1:]
        else:
            return patched_hidden_states

    def __enter__(self):
        target_module = self._get_target_module()
        self.hook = target_module.register_forward_hook(self._patch_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()

class AblationPatcher:
    """Context manager to ablate a transformer block by making it an identity function."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hook = None

    def ablation_hook(self, module, module_input, module_output):
        # Return input hidden states unchanged, making the layer an identity function
        return (module_input[0],) + module_output[1:]

    def __enter__(self):
        target_block = self.model.transformer.h[self.target_layer]
        self.hook = target_block.register_forward_hook(self.ablation_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()
