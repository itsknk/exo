from typing import Dict, Optional, Union
from tinygrad import Tensor, Variable, nn, Device
from .llama import Transformer as LlamaTransformer, sample, TransformerBlock, convert_from_huggingface
import math
import copy

# Vision MLP for the Vision Transformer
class VisionMLP:
    def __init__(self, dim: int, hidden_dim: int, linear=nn.Linear):
        self.fc1 = linear(dim, hidden_dim)
        self.fc2 = linear(hidden_dim, dim)
    
    def __call__(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x

# Vision Transformer Implementation
class VisionTransformer:
    def __init__(self, dim: int, n_layers: int, n_heads: int, hidden_dim: int, img_size: int = 224, patch_size: int = 14, in_channels: int = 3):
        self.dim = dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = Tensor.zeros(1, 1, dim, requires_grad=True)
        self.pos_embed = Tensor.zeros(1, self.n_patches + 1, dim, requires_grad=True)

        self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads=n_heads, norm_eps=1e-6, max_context=1024, feed_forward=VisionMLP, linear=nn.Linear) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(dim)
    
    def __call__(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # Shape: (bs, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # Shape: (bs, num_patches, dim)

        # Add class token
        cls_tokens = self.cls_token.expand(bs, -1, -1)  # Shape: (bs, 1, dim)
        x = Tensor.cat((cls_tokens, x), dim=1)  # Shape: (bs, num_patches + 1, dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Forward through transformer layers
        for layer in self.layers:
            x = layer(x, start_pos=0, freqs_cis=None, mask=None)

        # Apply layer norm to the class token
        x = self.norm(x[:, 0])  # Shape: (bs, dim)
        return x

# LLAVA Model Implementation
class LlavaModel:
    def __init__(self, vision_config: Dict, llama_config: Dict):
        self.vision_model = VisionTransformer(**vision_config)

        # Make a copy of llama_config to avoid modifying the original
        llama_args = copy.deepcopy(llama_config)

        # Extract special tokens
        self.image_token_id = llama_config.get('image_token_id', 32000)
        self.end_of_image_id = llama_config.get('end_of_image_id', 32001)
        self.eos_token_id = llama_config.get('eos_token_id', 2)

        # Remove special tokens and 'feed_forward' from llama_config before passing to LlamaTransformer
        exclude_keys = ['image_token_id', 'end_of_image_id', 'eos_token_id']
        if llama_config.get('feed_forward') is None:
            exclude_keys.append('feed_forward')
        llama_args = {k: v for k, v in llama_config.items() if k not in exclude_keys}

        # Initialize the language model with the corrected arguments
        self.language_model = LlamaTransformer(**llama_args)

        # **Set n_heads and n_kv_heads as attributes**
        self.language_model.n_heads = llama_args['n_heads']
        self.language_model.n_kv_heads = llama_args.get('n_kv_heads') or llama_args['n_heads']

        # Initialize the multimodal projector
        self.mm_projector = nn.Linear(vision_config['dim'], llama_args['dim'], bias=False)

    def __call__(
        self,
        tokens: Tensor,
        pixel_values: Optional[Tensor] = None,
        start_pos: Union[Variable, int] = 0,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        alpha_f: float = 0.0,
        alpha_p: float = 0.0,
    ):
        if pixel_values is not None:
            # Get image features from vision model
            vision_output = self.vision_model(pixel_values)  # Shape: (bs, dim)
            image_features = self.mm_projector(vision_output)  # Shape: (bs, dim)

            # Get token embeddings
            inputs_embeds = self.language_model.tok_embeddings(tokens)  # Shape: (bs, seq_len, dim)

            # Identify positions of image tokens
            image_token_mask = (tokens == self.image_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds * (~image_token_mask) + image_features.unsqueeze(1) * image_token_mask

            # Forward through language model using `inputs_embeds`
            logits = self.language_model.forward(
                inputs_embeds=inputs_embeds,
                start_pos=start_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                alpha_f=alpha_f,
                alpha_p=alpha_p,
            )
        else:
            # Directly pass tokens to the language model
            logits = self.language_model.forward(
                tokens=tokens,
                start_pos=start_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                alpha_f=alpha_f,
                alpha_p=alpha_p,
            )
        # Debugging statements
        print(f"tokens.shape: {tokens.shape}")
        if pixel_values is not None:
            print(f"vision_output.shape: {vision_output.shape}")
            print(f"image_features.shape: {image_features.shape}")
            print(f"inputs_embeds.shape: {inputs_embeds.shape}")
        else:
            print("No pixel_values provided")
        return logits

# Sampling Function (if needed)
def llava_sample(model: LlavaModel, tokens: Tensor, pixel_values: Optional[Tensor] = None, max_new_tokens: int = 50, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9, alpha_f: float = 0.0, alpha_p: float = 0.0):
    generated_tokens = tokens
    start_pos = 0
    for _ in range(max_new_tokens):
        logits = model(generated_tokens, pixel_values, start_pos, temperature, top_k, top_p, alpha_f, alpha_p)
        next_token = logits  # The language model's forward method already returns the sampled token

        # Append the new token
        generated_tokens = Tensor.cat((generated_tokens, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
        start_pos += 1

        # Check for end-of-sequence token
        if next_token.item() == model.language_model.eos_token_id:
            break
    return generated_tokens

# Function to convert weights from Hugging Face format to tinygrad model
from .llama import convert_from_huggingface
from tinygrad import Device

def convert_llava_from_huggingface(weights: Dict[str, Tensor], model: LlavaModel):
    # Map the weights from Hugging Face format to tinygrad model
    sd = {}
    for k, v in weights.items():
        v = v.to(Device.DEFAULT)
        if k.startswith("vision_model."):
            # Map vision model weights
            vision_key = k[len("vision_model."):]
            sd[f"vision_model.{vision_key}"] = v
        elif k.startswith("model."):
            # Map language model weights
            llama_key = k[len("model."):]
            # Use existing convert_from_huggingface function for llama
            llama_sd = convert_from_huggingface(
                {llama_key: v},
                model.language_model,
                model.language_model.n_heads,
                model.language_model.n_kv_heads,
            )
            # Adjust keys to include 'language_model.'
            sd.update({f"language_model.{k}": val for k, val in llama_sd.items()})
        elif k.startswith("mm_projector."):
            # Map multi-modal projector weights
            mm_proj_key = k[len("mm_projector."):]
            sd[f"mm_projector.{mm_proj_key}"] = v
        else:
            # Handle other keys if any
            pass
    return sd
