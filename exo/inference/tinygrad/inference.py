from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, convert_from_huggingface, fix_bf16
from exo.inference.tinygrad.models.llava import LlavaModel, convert_llava_from_huggingface
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import load_state_dict
from tinygrad import Tensor, nn, Context, Device
from exo.inference.inference_engine import InferenceEngine
from typing import Dict, Optional, Tuple
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
import asyncio
from safetensors import safe_open

Tensor.no_grad = True
# Default settings
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0

# Model parameters including LLAVA
MODEL_PARAMS = {
    "8B": {
        "args": {
            "dim": 4096,
            "n_heads": 32,
            "n_kv_heads": 8,
            "n_layers": 32,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 14336,
        },
        "files": 1,
    },
    "70B": {
        "args": {
            "dim": 8192,
            "n_heads": 64,
            "n_kv_heads": 8,
            "n_layers": 80,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 28672,
        },
        "files": 8,
    },
    "LLAVA": {
        "args": {
            "vision_config": {
                'dim': 768,
                'n_layers': 12,
                'n_heads': 12,
                'hidden_dim': 3072,
                'img_size': 224,
                'patch_size': 14,
                'in_channels': 3,
            },
            "llama_config": {
                'dim': 4096,
                'hidden_dim': 11008,
                'n_heads': 32,
                'n_layers': 32,
                'norm_eps': 1e-5,
                'vocab_size': 32064,
                'max_context': 2048,
                'rope_theta': 10000,
                'n_kv_heads': None,
                'jit': True,
                'feed_forward': None,  # Use default FeedForward
                'linear': nn.Linear,
                'image_token_id': 32000,  # Adjust based on the tokenizer?!
                'end_of_image_id': 32001,
                'eos_token_id': 2,  
            },
        },
        "files": 1,  # Adjust based on actual number of files for LLAVA?!
    },
}

def build_model(model_path: Path, shard: Shard, model_id: str, device=None):
    # Determine model type based on model_id
    if 'llava' in model_id.lower():
        model_type = 'LLAVA'
    elif 'llama' in model_id.lower():
        model_type = 'LLaMA'
    else:
        raise ValueError(f"Unknown model type for model_id: {model_id}")

    if model_type == 'LLAVA':
        # Build LLAVA model
        args = MODEL_PARAMS['LLAVA']['args']
        vision_config = args['vision_config']
        llama_config = args['llama_config']
        llama_config['shard'] = shard
        llama_config['linear'] = nn.Linear

        with Context(THREEFRY=0):
            model = LlavaModel(vision_config, llama_config)

        # Load weights
        weights = load_weights_for_llava(model_path, shard, model)
        with Context(BEAM=0):
            load_state_dict(model, weights, strict=False, consume=False)

    elif model_type == 'LLaMA':
        # Build LLaMA model as before
        model_size = "8B" if "8b" in model_id.lower() else "70B"
        linear = nn.Linear
        with Context(THREEFRY=0):
            model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True, shard=shard)

        # Load weights
        if model_path.is_dir():
            if (model_path / "model.safetensors.index.json").exists():
                weights = load(str(model_path / "model.safetensors.index.json"), shard)
            elif (model_path / "model.safetensors").exists():
                weights = load(str(model_path / "model.safetensors"), shard)
            else:
                weights = concat_weights(
                    [load(str(model_path / f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[model_size]["files"])],
                    device[0] if isinstance(device, tuple) else device,
                )
        else:
            weights = load(str(model_path), shard)

        weights = convert_from_huggingface(
            weights,
            model,
            MODEL_PARAMS[model_size]["args"]["n_heads"],
            MODEL_PARAMS[model_size]["args"]["n_kv_heads"],
        )
        weights = fix_bf16(weights)
        with Context(BEAM=0):
            # Replace weights in model
            load_state_dict(model, weights, strict=False, consume=False)

    return model

# Function to load weights for LLAVA model
def load_weights_for_llava(model_path, shard, model):
    # Check for .safetensors files
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    
    if safetensors_files:
        # Handle .safetensors files
        return load_safetensors_weights(model_path, shard, model)
    else:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

def load_safetensors_weights(model_path, shard, model):
    # Prepare the state dict
    state_dict = {}
    
    # Load the index file
    index_file = os.path.join(model_path, 'model.safetensors.index.json')
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file 'model.safetensors.index.json' not found in {model_path}")
    
    # Load the index
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    # Create a mapping from parameter names to files
    param_mappings = index['weight_map']
    
    # Iterate over all parameters
    for param_name, file_name in param_mappings.items():
        param_file = os.path.join(model_path, file_name)
        if not os.path.exists(param_file):
            raise FileNotFoundError(f"File {param_file} not found")
        
        # Open the safetensors file and load the parameter
        with safe_open(param_file, framework="numpy") as f:
            param_tensor = f.get_tensor(param_name)
            param_tensor = param_tensor.astype('float32')  # Ensure correct dtype
            param_tensor = Tensor(param_tensor).to(Device.DEFAULT)
            state_dict[param_name] = param_tensor
    
    # Convert state_dict to the format expected by the model
    converted_state_dict = convert_llava_from_huggingface(state_dict, model)
    return converted_state_dict

def convert_llava_from_huggingface(weights: Dict[str, Tensor], model: LlavaModel):
    # Map the weights from Hugging Face format to tinygrad model
    sd = {}
    for k, v in weights.items():
        v = v.to(Device.DEFAULT)
        if k.startswith("vision_model."):
            # Map vision model weights
            vision_key = k[len("vision_model."):]
            sd[f"vision_model.{vision_key}"] = v
        elif k.startswith("language_model."):
            # Map language model weights
            llama_key = k[len("language_model."):]
            # Use existing convert_from_huggingface function for llama
            llama_sd = convert_from_huggingface(
                {llama_key: v},
                model.language_model,
                model.language_model.n_heads,
                model.language_model.n_kv_heads or model.language_model.n_heads,
            )
            sd.update({f"language_model.{k}": val for k, val in llama_sd.items()})
        elif k.startswith("mm_projector."):
            # Map multi-modal projector weights
            mm_proj_key = k[len("mm_projector."):]
            sd[f"mm_projector.{mm_proj_key}"] = v
        else:
            # Handle other keys if any
            pass
    return sd

class TinygradDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def infer_prompt(
        self,
        request_id: str,
        shard: Shard,
        prompt: str,
        image_str: Optional[str] = None,
        inference_state: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

        if image_str:
            # Process the image string to obtain pixel_values
            pixel_values = await asyncio.get_event_loop().run_in_executor(self.executor, self.process_image, image_str)
        else:
            pixel_values = None

        toks = await asyncio.get_event_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
        input_tokens = Tensor([toks])

        # For LLAVA model, need to pass pixel_values
        h = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.model(
                input_tokens,
                pixel_values,
                start_pos,
                TEMPERATURE,
                TOP_K,
                TOP_P,
                ALPHA_F,
                ALPHA_P,
            ).realize(),
        )

        if h.shape == (1,):
            start_pos += len(toks)
            start_pos += 1
            n_captured_toks = 0
            return (
                np.array([[h.item()]]),
                json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}),
                h.item() == self.tokenizer.eos_token_id,
            )
        else:
            n_captured_toks = len(toks)
            return (
                h.numpy(),
                json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}),
                False,
            )

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

        h = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.model(
                Tensor(input_data),
                None,
                start_pos,
                TEMPERATURE,
                TOP_K,
                TOP_P,
                ALPHA_F,
                ALPHA_P,
            ).realize(),
        )

        if h.shape == (1,):
            start_pos += n_captured_toks
            start_pos += 1
            n_captured_toks = 0
            return (
                np.array([[h.item()]]),
                json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}),
                h.item() == self.tokenizer.eos_token_id,
            )
        else:
            return (
                h.numpy(),
                json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}),
                False,
            )

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return

        model_path = await self.shard_downloader.ensure_shard(shard)

        if self.shard != shard:
            self.model = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                build_model,
                model_path,
                shard,
                shard.model_id,
            )
            tokenizer_path = str(model_path if model_path.is_dir() else model_path.parent)
            self.tokenizer = await resolve_tokenizer(tokenizer_path)
            self.shard = shard

    def process_image(self, image_str: str) -> Tensor:
        # Convert image string to pixel_values tensor
        import base64
        from PIL import Image
        from io import BytesIO
        import numpy as np

        # Decode the base64 image string
        image_data = base64.b64decode(image_str)
        image = Image.open(BytesIO(image_data)).convert('RGB')

        # Resize and preprocess image
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0

        # Optionally normalize the image (mean and std should match training)?!
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_array = (image_array - mean) / std

        # Convert to Tensor and adjust dimensions
        pixel_values = Tensor(image_array).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 224, 224)
        return pixel_values
