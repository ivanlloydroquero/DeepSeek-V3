import os
import json
import logging
from argparse import ArgumentParser
from typing import List, Optional, Dict
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel
from transformers import AutoTokenizer, PreTrainedTokenizer
from safetensors.torch import load_model
from model import Transformer, ModelArgs

logger = logging.getLogger(__name__)

class GenerationConfig:
    """Modern generation configuration with essential parameters"""
    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        frequency_penalty: float = 0.1,
        stop_sequences: Optional[List[str]] = None,
        do_sample: bool = True,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.stop_sequences = stop_sequences or []
        self.do_sample = do_sample

def sample(
    logits: torch.Tensor,
    generation_config: GenerationConfig,
    past_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Modern sampling function with multiple decoding strategies
    
    Features:
    - Temperature scaling
    - Top-K + Top-P hybrid sampling
    - Repetition and frequency penalties
    - Efficient GPU operations
    """
    if past_tokens is not None and past_tokens.numel() > 0:
        if generation_config.repetition_penalty != 1.0:
            apply_repetition_penalty(logits, past_tokens, generation_config.repetition_penalty)
        
        if generation_config.frequency_penalty != 0.0:
            apply_frequency_penalty(logits, past_tokens, generation_config.frequency_penalty
                                    
    if generation_config.temperature > 0 and generation_config.do_sample:
        logits = logits / max(generation_config.temperature, 1e-5)
    else:
        return logits.argmax(dim=-1)

    if generation_config.top_k is not None and generation_config.top_k > 0:
        top_k = min(generation_config.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    if generation_config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > generation_config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, 
            index=sorted_indices, 
            src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def apply_repetition_penalty(
    logits: torch.Tensor,
    past_tokens: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """Advanced repetition penalty with token frequency scaling"""
    unique_tokens, counts = torch.unique(past_tokens, return_counts=True)
    score_ratio = torch.pow(penalty, counts.float() / counts.max())
    logits[..., unique_tokens] = logits[..., unique_tokens] / score_ratio
    return logits

def apply_frequency_penalty(
    logits: torch.Tensor,
    past_tokens: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """Frequency-based penalty using token occurrence counts"""
    unique, counts = torch.unique(past_tokens, return_counts=True)
    logits[..., unique] -= penalty * counts.float()
    return logits

@torch.inference_mode()
def generate(
    model: Transformer,
    tokenizer: PreTrainedTokenizer,
    prompts: List[List[int]],
    generation_config: GenerationConfig,
    device: torch.device,
) -> List[List[int]]:
    """
    Modern generation function with features:
    - Efficient attention computation with Flash Attention
    - KV cache management
    - Early stopping for stop sequences
    - Batch processing optimization
    - Sequence repetition prevention
    """
    batch_size = len(prompts)
    max_seq_length = model.max_seq_len
    eos_id = tokenizer.eos_token_id

    tokens = torch.full((batch_size, max_seq_length), -1, dtype=torch.long, device=device)
    for i, seq in enumerate(prompts):
        tokens[i, :len(seq)] = torch.tensor(seq, device=device)

    model.init_kv_cache(batch_size, max_seq_length)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    prev_pos = 0

    attn_ctx = sdpa_kernel() if hasattr(torch.nn.attention, 'sdpa_kernel') else nullcontext()

    with attn_ctx:
        for cur_pos in range(len(prompts[0]), max_seq_length):
            if finished.all():
                break

            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_tokens = sample(
                logits[:, -1],
                generation_config,
                tokens[:, :cur_pos] if cur_pos > 0 else None
            )

            tokens[:, cur_pos] = torch.where(finished, eos_id, next_tokens)
            finished = finished | (next_tokens == eos_id)
            
            if generation_config.stop_sequences:
                decoded = tokenizer.batch_decode(tokens[:, :cur_pos+1])
                for i, text in enumerate(decoded):
                    if any(seq in text for seq in generation_config.stop_sequences):
                        finished[i] = True

            prev_pos = cur_pos

    return [seq[seq != -1].tolist() for seq in tokens]

def load_model_distributed(
    model: Transformer,
    ckpt_path: str,
    rank: int,
    world_size: int
) -> None:
    """Optimized model loading with distributed support"""
    ckpt_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if os.path.exists(ckpt_file):
        load_model(model, ckpt_file)
    else:
        raise FileNotFoundError(f"Checkpoint file {ckpt_file} not found")
    
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param, src=0)

def main(
    ckpt_path: str,
    config_path: str,
    input_file: str = "",
    interactive: bool = True,
    generation_config: GenerationConfig = GenerationConfig(),
) -> None:
    """Enhanced main function with modern features"""

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.enable_flash_sdp(True)
    
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))
    
    model = Transformer(model_args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model_distributed(model, ckpt_path, rank, world_size)
    
    if interactive:
        chat_interface(model, tokenizer, generation_config, device, rank, world_size)
    else:
        batch_processing(model, tokenizer, input_file, generation_config, device, rank)

    if world_size > 1:
        dist.destroy_process_group()

def chat_interface(
    model: Transformer,
    tokenizer: PreTrainedTokenizer,
    config: GenerationConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Modern chat interface with conversation history"""
    messages = []
    while True:
        try:
            prompt = distributed_input(">>> ", rank, world_size)
            if prompt.lower() == "/exit":
                break
            if prompt.lower() == "/reset":
                messages.clear()
                continue
            
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = [tokenizer.apply_chat_template(messages, add_generation_prompt=True)]
            
            output_tokens = generate(
                model,
                tokenizer,
                prompt_tokens,
                config,
                device
            )
            
            response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            distributed_print(f"<<< {response}", rank)
            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break

def distributed_input(prompt: str, rank: int, world_size: int) -> str:
    """Handle distributed input collection"""
    if world_size == 1 or rank == 0:
        user_input = input(prompt)
        if world_size > 1:
            objs = [user_input]
            dist.broadcast_object_list(objs, src=0)
        return user_input
    else:
        objs = [None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

def distributed_print(text: str, rank: int) -> None:
    """Ensure clean printing in distributed environment"""
    if rank == 0:
        print(text)

if __name__ == "__main__":
    parser = ArgumentParser(description="Modern Distributed Text Generation")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    main(
        ckpt_path=args.ckpt_path,
        config_path=args.config,
        input_file=args.input_file,
        interactive=args.interactive,
        generation_config=generation_config,
    )
