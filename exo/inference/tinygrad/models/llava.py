from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad import dtypes
import math
from typing import Optional, Tuple

class RotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq
        self.max_seq_len_cached = max_position_embeddings
        t = Tensor.arange(self.max_seq_len_cached).cast(dtypes.float32)
        freqs = Tensor.outer(t, self.inv_freq)
        emb = Tensor.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = Tensor.arange(self.max_seq_len_cached).cast(dtypes.float32)
        freqs = Tensor.outer(t, self.inv_freq)
        emb = Tensor.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

class LLaMARotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__(dim, max_position_embeddings, base)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return Tensor.cat((-x2, x1), dim=-1)

    def forward(self, query_layer: Tensor, key_layer: Tensor, value_layer: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        seq_len = key_layer.shape[-2]
        cos, sin = super().forward(query_layer, seq_len=seq_len)
        query_layer, key_layer = query_layer.float(), key_layer.float()
        query_layer = query_layer.reshape(*query_layer.shape[:-1], -1, 2)
        key_layer = key_layer.reshape(*key_layer.shape[:-1], -1, 2)
        query_layer = Tensor.cat([-query_layer[..., 1::2], query_layer[..., ::2]], axis=-1)
        key_layer = Tensor.cat([-key_layer[..., 1::2], key_layer[..., ::2]], axis=-1)
        query_layer, key_layer = query_layer * cos + self.rotate_half(query_layer) * sin, key_layer * cos + self.rotate_half(key_layer) * sin
        return query_layer.cast(value_layer.dtype), key_layer.cast(value_layer.dtype), value_layer

class LLaMAAttention:
    def __init__(self, hidden_size, num_heads, rotary_dim):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = rotary_dim

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = LLaMARotaryEmbedding(self.rotary_dim)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None, past_key_value: Optional[Tuple[Tensor, Tensor]] = None, inputs_embeds: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states, value_states = self.rotary_emb.forward(query_states, key_states, value_states)

        if past_key_value is not None:
            key_states = Tensor.cat([past_key_value[0], key_states], dim=2)
            value_states = Tensor.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        attn_weights = Tensor.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = Tensor.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class LLaMAMLP:
    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj((self.gate_proj(x).sigmoid() * self.up_proj(x)))

class LLaMADecoderLayer:
    def __init__(self, hidden_size, num_heads, intermediate_size, rotary_dim):
        self.hidden_size = hidden_size
        self.self_attn = LLaMAAttention(hidden_size=self.hidden_size, num_heads=num_heads, rotary_dim=rotary_dim)
        self.mlp = LLaMAMLP(hidden_size=self.hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm = nn.LayerNorm(self.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None, past_key_value: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, past_key_value=past_key_value)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class LLaMAModel:
    def __init__(self, config):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = [LLaMADecoderLayer(config.hidden_size, config.num_attention_heads, config.intermediate_size, config.rotary_dim) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, input_ids: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None, past_key_values: Optional[Tuple[Tuple[Tensor]]] = None, inputs_embeds: Optional[Tensor] = None):
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            hidden_states, _ = decoder_layer(hidden_states, attention_mask=attention_mask, past_key_value=past_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states

class LLaMAForCausalLM:
    def __init__(self, config):
        self.model = LLaMAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None, past_key_values: Optional[Tuple[Tuple[Tensor]]] = None, inputs_embeds: Optional[Tensor] = None):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds)
        logits = self.lm_head(hidden_states)
        return logits

class CLIPVisionEmbeddings:
    def __init__(self, config):
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = Tensor.randn(self.embed_dim)

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = Tensor.arange(self.num_positions).expand((1, -1))

    def __call__(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.reshape(batch_size, self.embed_dim, -1).transpose(1, 2)

        class_embeds = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        embeddings = Tensor.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class CLIPEncoderLayer:
    def __init__(self, config):
        self.embed_dim = config.hidden_size
        self.self_attn = LLaMAAttention(hidden_size=self.embed_dim, num_heads=config.num_attention_heads, rotary_dim=config.rotary_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = LLaMAMLP(hidden_size=self.embed_dim, intermediate_size=config.intermediate_size)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def __call__(self, hidden_states):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class CLIPVisionTransformer:
    def __init__(self, config):
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states

class CLIPVisionModel:
    def __init__(self, config):
        self.config = config
        self.vision_model = CLIPVisionTransformer(config)

    def __call__(self, pixel_values):
        return self.vision_model(pixel_values)

class VisionConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class TextConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class LLaVAConfig:
    def __init__(self, vision_config: VisionConfig, text_config: TextConfig):
        self.vision_config = vision_config
        self.text_config = text_config

class LLaVAModel:
    def __init__(self, config: LLaVAConfig):
        self.config = config

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.language_model = LLaMAForCausalLM(config.text_config)

        self.vision_projection = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)

    def __call__(self, pixel_values, input_ids, attention_mask=None):
        vision_outputs = self.vision_model(pixel_values)
        image_embeds = vision_outputs[:, 0, :]  # Take CLS token
        image_embeds = self.vision_projection(image_embeds)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        inputs_embeds = Tensor.cat([image_embeds.unsqueeze(1), inputs_embeds], dim=1)

        if attention_mask is not None:
            attention_mask = Tensor.cat([Tensor.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype), attention_mask], dim=1)

        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs

def create_llava_model(vision_config, text_config):
    # Print the type of vision_config for debugging
    print(f"vision_config is of type: {type(vision_config)}")

    if isinstance(vision_config, VisionConfig):
        print("vision_config is an instance of VisionConfig")
        # Use the VisionConfig object directly (no need to convert to dict)
        vision_config_instance = vision_config  # Keep it as a VisionConfig instance
    elif isinstance(vision_config, dict):
        print("vision_config is a dictionary")
        # Convert the dict to a VisionConfig instance
        vision_config_instance = VisionConfig(**vision_config)
    else:
        raise TypeError("Expected a dict or VisionConfig instance for vision_config")

    # Proceed with creating the llava model using the vision_config_instance
    print("Creating the llava model with vision_config...")


# Now use vision_config as an instance of VisionConfig

    if not isinstance(text_config, TextConfig):
        text_config = TextConfig(**text_config)
    
    vision_config.rotary_dim = vision_config.hidden_size
    text_config.rotary_dim = text_config.hidden_size
    
    config = LLaVAConfig(vision_config=vision_config, text_config=text_config)
    return LLaVAModel(config)
