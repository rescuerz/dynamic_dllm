import os
import argparse
import logging
import random
import json
from typing import Dict, Union, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer

# 导入LLaDA模型
try:
    from model.modeling_llada import LLaDAModelLM
except ImportError:
    from llada.model.modeling_llada import LLaDAModelLM

# 导入数据集处理模块
from dynamic_length_dataset import DynamicLengthDataset, DynamicLengthConfig, SPECIAL_TOKENS

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 设置tokenizer并行处理，避免多进程冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"




def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLaDA扩散模型动态长度微调')
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                       help='基础模型名称')
    parser.add_argument('--data_path', type=str, required=True,
                       help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批次大小（8B模型建议使用较小的batch size）')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='学习率（8B模型建议使用较小的学习率）')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='最大训练步数')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='预热步数')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='保存间隔步数')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='日志记录间隔')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='梯度累积步数（8B模型需要更多梯度累积）')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='最大序列长度')
    parser.add_argument('--dtype', type=str, default='float16',
                       help='模型数据类型（bfloat16/float16/float32）')
    parser.add_argument('--trust_remote_code', action='store_true', default=True,
                       help='是否信任远程代码（LLaDA模型需要）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    args = parser.parse_args()
    return args


def setup_model_and_tokenizer(args):
    """设置模型和tokenizer"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        use_fast=True,
        trust_remote_code=args.trust_remote_code
    )

    # 添加特殊token
    special_tokens = list(SPECIAL_TOKENS.values())
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in special_tokens if token not in existing_tokens]

    if new_tokens:
        special_tokens_dict = {'additional_special_tokens': new_tokens}
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added} new special tokens: {new_tokens}")

    # 设置pad token
    if tokenizer.pad_token is None:
        # 检查pad token是否存在
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Pad token is None, set to eos token: {tokenizer.eos_token}")
    logger.info(f"Pad token: {tokenizer.pad_token}")
    logger.info(f"Pad token id: {tokenizer.pad_token_id}")
    logger.info(f"Eos token: {tokenizer.eos_token}")
    logger.info(f"Eos token id: {tokenizer.eos_token_id}")
    

    logger.info(f"Mask token: {tokenizer.mask_token}")

    # 先打印tokenizer的mask token信息
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.info(f"Tokenizer mask token: {getattr(tokenizer, 'mask_token', 'None')}")
    logger.info(f"Tokenizer mask token id: {getattr(tokenizer, 'mask_token_id', 'None')}")
    logger.info(f"Mask token id (from tokenizer only): {get_mask_token_id(tokenizer)}")

    # 加载模型
    logger.info(f"Loading model from {args.model_name}")
    model = LLaDAModelLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
    )

    # 模型加载后，打印模型配置中的mask token信息
    logger.info(f"Model config mask_token_id: {getattr(model.config, 'mask_token_id', 'None')}")
    logger.info(f"Model config vocab_size: {getattr(model.config, 'vocab_size', 'None')}")
    logger.info(f"Model config embedding_size: {getattr(model.config, 'embedding_size', 'None')}")
    logger.info(f"Model config eos_token_id: {getattr(model.config, 'eos_token_id', 'None')}")
    logger.info(f"Model config pad_token_id: {getattr(model.config, 'pad_token_id', 'None')}")

    # 获取最终的mask token ID（优先使用模型配置）
    final_mask_token_id = get_mask_token_id(tokenizer, model)
    logger.info(f"Final mask token id (with model config): {final_mask_token_id}")

    # 如果添加了新token，需要调整embedding层
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")

    return model, tokenizer


def get_enlarge_token_ids(tokenizer: AutoTokenizer) -> Dict[str, int]:
    """
    获取扩展token的ID映射
    """
    enlarge_token_ids = {}
    for key, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            enlarge_token_ids[key] = token_id

    logger.info(f"Enlarge token IDs: {enlarge_token_ids}")
    return enlarge_token_ids


def get_mask_token_id(tokenizer: AutoTokenizer, model=None) -> int:
    """
    获取mask token ID
    """
    # 方法1：优先从模型配置获取（LLaDA模型的标准做法）
    if model is not None and hasattr(model, 'config') and hasattr(model.config, 'mask_token_id'):
        if model.config.mask_token_id is not None:
            logger.info(f"Using mask_token_id from model config: {model.config.mask_token_id}")
            return model.config.mask_token_id

    # 方法2：如果没有模型，使用LLaDA的标准mask_token_id
    llada_default_mask_id = 126336  # LLaDA模型的标准mask_token_id
    logger.warning(f"Model not available, using LLaDA default mask_token_id: {llada_default_mask_id}")
    return llada_default_mask_id


def forward_diffusion_process_with_dynamic_length(input_ids, mask_token_id, eps=1e-3,
                                                 current_length=None, special_token_ids=None,
                                                 timestep=None, config=None):
    """
    支持动态长度的前向扩散过程，将输入序列的部分token随机替换为mask token，模拟扩散过程中的噪声添加。

    Args:
        input_ids: 输入token序列（response部分） [batch_size, seq_len]
        mask_token_id: mask token的ID，用于替换被mask的位置
        eps: 最小mask概率
        current_length: 当前有效长度(response 长度)（支持动态长度）
        special_token_ids: 特殊token ID映射（用于后续检查）
        timestep: 指定的时间步，如果为None则随机采样
        config: 动态长度配置对象

    Returns:
        tuple: (noisy_input, p_mask, mask_indices, actual_timestep)
    """
    b, l = input_ids.shape
    device = input_ids.device

    # 动态长度处理，从短序列开始，逐步扩展到长序列
    if current_length is not None:
        effective_length = min(current_length, l)
    else:
        effective_length = l

    # 时间步处理：可以指定或随机采样
    if timestep is not None:
        # 使用指定的时间步
        if isinstance(timestep, (int, float)):
            t = torch.full((b,), timestep, device=device)
        else:
            t = timestep
    else:
        # 随机采样时间步
        t = torch.rand((b,), device=device)

    # 计算mask概率
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, effective_length)

    # 创建完整的mask概率张量
    full_p_mask = torch.zeros((b, l), device=device)
    full_p_mask[:, :effective_length] = p_mask

    # 随机mask
    mask_indices = torch.rand((b, effective_length), device=device) < p_mask

    # Special token与普通token一视同仁，不需要额外保护
    # 这样可以让模型学会在适当的位置生成special token

    # 生成噪声输入
    noisy_input = input_ids.clone()
    noisy_input[:, :effective_length] = torch.where(
        mask_indices,
        mask_token_id,  # 使用mask_token_id作为mask token
        input_ids[:, :effective_length]
    )

    return noisy_input, full_p_mask, mask_indices, t


def train_single_sample_at_length(input_ids, prompt_length, current_length, model, loss_func,
                                special_token_ids, device, config, tokenizer, mask_token_id):
    """
    对单个样本在指定长度进行扩散训练，只对response部分进行扩散训练，保持prompt部分不变

    Returns:
        tuple: (loss, mask_indices) - 损失值和mask位置信息
    """
    # 截取到当前长度
    current_input = input_ids[:, :current_length]

    # 计算response长度
    prompt_len = prompt_length[0].item()
    response_length = max(1, current_length - prompt_len)

    # 随机采样解码比例
    detection_min = config.expansion_check_ratio - 0.05
    detection_max = config.expansion_check_ratio + 0.05
    target_decoded_ratio = torch.rand(1).item() * (detection_max - detection_min) + detection_min

    # 计算mask参数
    target_decoded_tokens = int(response_length * target_decoded_ratio)
    target_mask_tokens = response_length - target_decoded_tokens
    mask_ratio = target_mask_tokens / response_length

    # 计算时间步（已知mask_ratio，解出timestep）
    eps = 1e-3
    timestep = max(0.0, min(1.0, (mask_ratio - eps) / (1 - eps)))

    logger.info(f"Single sample training: prompt_len={prompt_len}, response_length={response_length}, "
                f"decode_ratio={target_decoded_ratio:.2f}, mask_ratio={mask_ratio:.2f}, timestep={timestep:.2f}")

    # 只对response部分应用扩散
    noisy_input = current_input.clone()
    mask_indices = torch.zeros_like(current_input, dtype=torch.bool)  # 初始化mask indices

    if response_length > 0:
        response_part = current_input[0, prompt_len:prompt_len + response_length].unsqueeze(0)
        noisy_response, _, response_mask_indices, _ = forward_diffusion_process_with_dynamic_length(
            response_part, mask_token_id=mask_token_id,
            current_length=response_length, special_token_ids=special_token_ids,
            timestep=timestep, config=config
        )
        noisy_input[0, prompt_len:prompt_len + response_length] = noisy_response[0]
        # 记录response部分的mask信息
        mask_indices[0, prompt_len:prompt_len + response_length] = response_mask_indices[0]

    # 创建p_mask，用于损失计算的重加权
    p_mask = torch.zeros_like(noisy_input, dtype=torch.float)
    if response_length > 0:
        response_mask_prob = (1 - eps) * timestep + eps
        p_mask[0, prompt_len:prompt_len + response_length] = response_mask_prob

    # 模型前向传播
    actual_mask_indices = (noisy_input == mask_token_id)
    mask_count = actual_mask_indices.sum().item()
    logger.debug(f"Forward pass: mask_count={mask_count}, input_shape={noisy_input.shape}")

    # 构建attention_mask：在训练阶段可选择性排除special token
    attention_mask = torch.ones_like(noisy_input, dtype=torch.long)

    # 新增配置：是否在训练时也排除special token attention
    exclude_in_training = getattr(config, 'exclude_special_tokens_in_training', False)

    if config.exclude_special_tokens_from_attention and special_token_ids and exclude_in_training:
        for token_name, token_id in special_token_ids.items():
            if token_id is not None:
                # 将special token位置的attention_mask设为0
                special_positions = (noisy_input == token_id)
                if special_positions.any():
                    attention_mask = attention_mask & (~special_positions)
                    logger.debug(f"Excluded special token '{token_name}' (id={token_id}) from attention at {special_positions.sum().item()} positions")

        # 额外检查：确保原始输入中的special token也被排除
        # 考虑special token可能未被mask的情况
        for token_name, token_id in special_token_ids.items():
            if token_id is not None:
                original_special_positions = (current_input == token_id)
                if original_special_positions.any():
                    attention_mask = attention_mask & (~original_special_positions)
                    logger.debug(f"Excluded original special token '{token_name}' (id={token_id}) from attention at {original_special_positions.sum().item()} positions")

    outputs = model(input_ids=noisy_input, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

    # 计算损失
    if actual_mask_indices.any():
        loss = loss_func(logits[actual_mask_indices], current_input[actual_mask_indices], reduction='none') / p_mask[actual_mask_indices]
        loss = loss.sum() / response_length
        logger.debug(f"Loss calculation: masked_positions_loss={loss.item():.4f}")
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        logger.debug("Loss calculation: no masked positions, loss=0")

    return loss, mask_indices


def check_unmasked_special_tokens(input_ids, mask_positions, special_token_ids, current_response_length, prompt_length):
    """
    检查未被mask的special token，直接进行扩展决策

    Args:
        input_ids: 输入序列 [batch_size, seq_len]
        mask_positions: mask位置 [batch_size, seq_len] 布尔张量
        special_token_ids: 特殊token ID映射
        current_response_length: 当前response的训练长度（不包括prompt）
        prompt_length: 每个样本的prompt长度

    Returns:
        dict or None: 扩展决策字典，如果没有直接决策则返回None
    """
    batch_size = input_ids.size(0)

    enlarge_token_id = special_token_ids.get('enlarge')
    enough_token_id = special_token_ids.get('enough')

    # 定义special token应该出现的位置（0基索引）
    special_token_positions = [63, 127, 255, 511, 1023, 2047]

    for i in range(batch_size):
        # 计算response部分的范围
        if isinstance(prompt_length, torch.Tensor):
            sample_prompt_len = prompt_length[i].item() if len(prompt_length.shape) > 0 else prompt_length.item()
        else:
            sample_prompt_len = prompt_length

        response_start = sample_prompt_len

        # 找到当前response长度范围内的special token位置
        valid_positions = []
        for pos in special_token_positions:
            if pos < current_response_length:  # 只检查当前response长度范围内的位置
                absolute_pos = response_start + pos
                if absolute_pos < input_ids.size(1):  # 确保不超出序列边界
                    valid_positions.append((pos, absolute_pos))

        if not valid_positions:
            continue  # 当前response长度范围内没有special token位置

        # 关键修正：检查最大位置的special token（最接近当前扩展边界）
        valid_positions.sort(key=lambda x: x[0], reverse=True)
        max_pos, max_absolute_pos = valid_positions[0]

        logger.debug(f"Sample {i}: Checking max position {max_pos} (absolute {max_absolute_pos}) within response length {current_response_length}")

        # 只检查最大位置的token
        if not mask_positions[i, max_absolute_pos]:
            token_id = input_ids[i, max_absolute_pos].item()

            if token_id == enlarge_token_id:
                # 发现未被mask的enlarge token，直接决策扩展
                logger.debug(f"Sample {i}: Found unmasked <enlarge> at max position {max_absolute_pos} (relative {max_pos}), direct expand")
                return {
                    'expand': True,
                    'confidence': 1.0,  # 直接观察到，置信度最高
                    'method': 'direct_unmasked',
                    'absolute_position': max_absolute_pos,
                    'relative_position': max_pos,
                    'token_type': 'enlarge'
                }

            elif token_id == enough_token_id:
                # 发现未被mask的enough token，直接决策停止
                logger.debug(f"Sample {i}: Found unmasked <enough> at max position {max_absolute_pos} (relative {max_pos}), direct stop")
                return {
                    'expand': False,
                    'confidence': 1.0,  # 直接观察到，置信度最高
                    'method': 'direct_unmasked',
                    'absolute_position': max_absolute_pos,
                    'relative_position': max_pos,
                    'token_type': 'enough'
                }

    # 没有找到直接决策
    return None





def detect_sample_expansion(input_ids, prompt_length, current_length, model, special_token_ids, config, mask_token_id):
    """
    检测单个样本是否需要扩展，检查当前response长度范围内最大位置的special token概率

    Args:
        input_ids: 输入序列 [batch_size, seq_len]
        prompt_length: prompt长度
        current_length: 当前总序列长度（prompt + response）

    Returns:
        dict: 扩展决策字典
    """
    current_input = input_ids[:, :current_length]

    # 计算当前response长度
    prompt_len = prompt_length[0].item() if isinstance(prompt_length, torch.Tensor) else prompt_length
    current_response_length = current_length - prompt_len

    # 定义special token应该出现的位置（0基索引，相对于response开始位置）
    special_token_positions = [63, 127, 255, 511, 1023, 2047]

    # 找到当前response长度范围内的最大special token位置
    max_special_pos = None
    for pos in reversed(special_token_positions):  # 从大到小检查
        if pos < current_response_length:
            max_special_pos = pos
            break

    if max_special_pos is None:
        logger.info(f"No special token position within current response length {current_response_length}")
        return {'expand': False, 'confidence': 0.0, 'reason': 'no_special_position'}

    # 计算绝对位置
    max_absolute_pos = prompt_len + max_special_pos

    logger.info(f"Checking special token prediction at response position {max_special_pos} (absolute {max_absolute_pos}) within response length {current_response_length}")

    # 构建attention_mask：排除special token
    attention_mask = torch.ones_like(current_input, dtype=torch.long)
    if config.exclude_special_tokens_from_attention and special_token_ids:
        for token_name, token_id in special_token_ids.items():
            if token_id is not None:
                special_positions = (current_input == token_id)
                if special_positions.any():
                    attention_mask = attention_mask & (~special_positions)
                    logger.info(f"Expansion detection: excluded special token '{token_name}' from attention at {special_positions.sum().item()} positions")

    # 模型前向传播
    outputs = model(input_ids=current_input, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

    # 检查最大special token位置的概率
    if max_absolute_pos < logits.size(1):
        position_logits = logits[0, max_absolute_pos, :]
        probabilities = torch.softmax(position_logits, dim=-1)

        enlarge_token_id = special_token_ids.get('enlarge')
        enough_token_id = special_token_ids.get('enough')

        # 记录所有特殊token的概率
        token_probs = {}
        for token_name, token_id in special_token_ids.items():
            if token_id is not None and token_id < logits.size(-1):
                token_probs[token_name] = probabilities[token_id].item()

        logger.info(f"Model prediction at position {max_absolute_pos}: special_token_probs={token_probs}")

        # 检查enlarge token概率
        if enlarge_token_id is not None and enlarge_token_id < logits.size(-1):
            enlarge_prob = probabilities[enlarge_token_id].item()
            if enlarge_prob > config.confidence_threshold:
                logger.info(f"Model predicts <enlarge> token, prob={enlarge_prob:.4f} > threshold={config.confidence_threshold}")
                return {'expand': True, 'confidence': enlarge_prob, 'method': 'model_prediction', 'position': max_absolute_pos}

        # 检查enough token概率
        if enough_token_id is not None and enough_token_id < logits.size(-1):
            enough_prob = probabilities[enough_token_id].item()
            if enough_prob > config.confidence_threshold:
                logger.info(f"Model predicts <enough> token, prob={enough_prob:.4f} > threshold={config.confidence_threshold}")
                return {'expand': False, 'confidence': enough_prob, 'reason': 'enough', 'method': 'model_prediction', 'position': max_absolute_pos}

    logger.info(f"Model prediction confidence too low, stopping expansion")
    return {'expand': False, 'confidence': 0.0, 'reason': 'low_confidence', 'method': 'model_prediction'}



def get_next_response_expansion_length(current_response_length, response_expansion_steps):
    """
    获取下一个response扩展长度
    """
    for step in response_expansion_steps:
        if step > current_response_length:
            return step
    return current_response_length  # 无法扩展




def train_dynamic_diffusion_step_multi_expansion(input_ids, prompt_length, model, loss_func,
                                               special_token_ids, device, config, tokenizer):
    """
    样本级独立动态扩展的扩散训练

    核心改进：
    1. 训练目标：response长度而不是整个序列长度
    2. 扩展步骤：64 -> 128 -> 256 -> 512 -> 1024 -> 2048 (response长度)
    3. 每个样本独立决定是否需要扩展
    4. 检查逻辑基于当前response长度范围内的最大special token位置
    """
    # 动态获取mask token ID
    mask_token_id = get_mask_token_id(tokenizer, model)

    # Response长度扩展步骤（核心修正）
    response_expansion_steps = [64, 128, 256, 512, 1024, 2048]
    initial_response_length = response_expansion_steps[0]  # 从64开始
    max_expansions = config.max_expansions

    batch_size = input_ids.shape[0]
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 样本级状态跟踪（修正：存储response长度而不是整个序列长度）
    sample_current_response_lengths = [initial_response_length] * batch_size
    sample_active = [True] * batch_size  # 哪些样本还需要继续训练
    sample_losses = []  # 记录每个样本的损失
    training_rounds = 0

    logger.info(f"Starting response-based training: batch_size={batch_size}, initial_response_length={initial_response_length}, max_expansions={max_expansions}")

    # 样本级独立训练循环
    while any(sample_active) and training_rounds < max_expansions:
        training_rounds += 1
        active_samples = [i for i, active in enumerate(sample_active) if active]

        if not active_samples:
            break

        logger.info(f"Training round {training_rounds}/{max_expansions}: active_samples={len(active_samples)}/{batch_size}")

        # 对每个活跃样本进行独立训练
        for sample_idx in active_samples:
            current_response_length = sample_current_response_lengths[sample_idx]

            # 计算实际可用的response长度
            prompt_len = prompt_length[sample_idx].item() if isinstance(prompt_length, torch.Tensor) else prompt_length
            max_possible_response_length = input_ids.size(1) - prompt_len

            # 确保不超过实际可用长度
            if current_response_length > max_possible_response_length:
                sample_active[sample_idx] = False
                logger.info(f"Sample {sample_idx}: response length {current_response_length} exceeds max possible {max_possible_response_length}, stopping")
                continue

            # 计算当前整个序列的训练长度
            current_total_length = prompt_len + current_response_length

            # 单样本训练（修正：传入整个序列长度，但内部基于response长度处理）
            sample_loss, mask_indices = train_single_sample_at_length(
                input_ids[sample_idx:sample_idx+1],  # 单个样本
                prompt_length[sample_idx:sample_idx+1],
                current_total_length,  # 整个序列的当前训练长度
                model, loss_func, special_token_ids, device, config, tokenizer,
                mask_token_id
            )

            # 首先检查未被mask的special token，进行直接决策（修正：传入response长度）
            direct_decision = check_unmasked_special_tokens(
                input_ids[sample_idx:sample_idx+1],
                mask_indices,
                special_token_ids,
                current_response_length,  # 修正：传入response长度而不是整个序列长度
                prompt_length[sample_idx:sample_idx+1]
            )

            if direct_decision is not None:
                # 找到了未被mask的special token，直接使用决策
                expansion_decision = direct_decision
                logger.info(f"Sample {sample_idx}: Direct decision from unmasked special token: {expansion_decision}")
            else:
                # 没有找到直接决策，使用模型预测来判断是否扩展
                logger.info(f"Sample {sample_idx}: No direct decision, using model prediction for expansion")
                expansion_decision = detect_sample_expansion(
                    input_ids[sample_idx:sample_idx+1],
                    prompt_length[sample_idx:sample_idx+1],
                    current_total_length,
                    model,
                    special_token_ids,
                    config,
                    mask_token_id
                )
                logger.info(f"Sample {sample_idx}: Model prediction result: {expansion_decision}")

            # 累加损失
            total_loss = total_loss + sample_loss
            sample_losses.append(sample_loss.item())

            logger.info(f"Sample {sample_idx}: response_length={current_response_length}, loss={sample_loss.item():.4f}, expansion={expansion_decision}")

            # 根据扩展决策更新样本状态（修正：基于response长度扩展）
            if expansion_decision['expand']:
                # 需要扩展，更新到下一个response长度
                next_response_length = get_next_response_expansion_length(current_response_length, response_expansion_steps)
                if next_response_length > current_response_length and next_response_length <= max_possible_response_length:
                    sample_current_response_lengths[sample_idx] = next_response_length
                    logger.info(f"Sample {sample_idx}: expanding response from {current_response_length} to {next_response_length}")
                else:
                    # 无法再扩展，标记为完成
                    sample_active[sample_idx] = False
                    logger.info(f"Sample {sample_idx}: cannot expand response further, current={current_response_length}, stopping")
            else:
                # 内容已足够，停止该样本的训练
                sample_active[sample_idx] = False
                logger.info(f"Sample {sample_idx}: response content sufficient, stopping")

    # 损失归一化
    if sample_losses:
        avg_loss = sum(sample_losses) / len(sample_losses)
        total_loss = total_loss / len(sample_losses)
        logger.info(f"Sample-level training complete: rounds={training_rounds}, sample_losses={len(sample_losses)}, avg_loss={avg_loss:.4f}")
    else:
        logger.warning("Sample-level training complete, but no losses were computed")

    return total_loss

def save_checkpoint(model, tokenizer, optimizer, scheduler, step, output_dir, is_final=False):
    """保存检查点"""
    if is_final:
        checkpoint_dir = output_dir / "final_model"
    else:
        checkpoint_dir = output_dir / f"checkpoint-{step}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型和tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # 保存训练状态
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
    }, checkpoint_dir / "training_state.pt")

    logger.info(f"Saved checkpoint to {checkpoint_dir}")




def train_dynamic_length_sft(args):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # 移动模型到设备
    model.to(device)

    # 创建配置
    config = DynamicLengthConfig()

    # 获取扩展token的ID映射
    enlarge_token_ids = get_enlarge_token_ids(tokenizer)

    # 创建数据集
    logger.info("Loading dataset...")
    dataset = DynamicLengthDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        config=config,
    )

    # 打印数据集统计信息
    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics: {stats}")

    # 启用special token排除日志记录（仅在调试时）
    if logger.level <= logging.DEBUG:
        dataset.enable_special_token_logging(True)

    # 创建数据加载器
    def collate_fn(batch):
        """数据整理函数"""
        # 计算最大长度
        max_len = max(len(item["input_ids"]) for item in batch)

        # 批量处理padding
        input_ids = []
        attention_mask = []
        labels = []
        prompt_lengths = []
        enlarge_token_ids = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            prompt_lengths.append(item["prompt_length"])
            enlarge_token_ids.append(item["enlarge_token_id"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "enlarge_token_ids": torch.tensor(enlarge_token_ids, dtype=torch.long),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=True
    )

    # 设置优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # 计算总步数
    total_steps = min(args.max_steps, len(dataloader) * args.num_epochs)

    # 设置学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate * 0.1
    )

    # 训练循环
    model.train()
    global_step = 0
    total_loss = 0.0

    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info(f"Total steps: {total_steps}, Batch size: {args.batch_size}, Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info("=" * 50)

    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        epoch_loss = 0.0
        epoch_samples = 0
        
        for step, batch in enumerate(dataloader):
            if global_step >= args.max_steps:
                break

            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            batch_size = batch["input_ids"].size(0) if "input_ids" in batch else 0
            logger.info(f"Batch {step+1}: samples={batch_size}, global_step={global_step+1}/{total_steps}")

            # 执行多次扩展的动态长度扩散训练步骤
            logger.info("Starting dynamic diffusion training for current batch")
            loss = train_dynamic_diffusion_step_multi_expansion(
                input_ids=batch["input_ids"],
                prompt_length=batch["prompt_lengths"],
                model=model,
                loss_func=torch.nn.functional.cross_entropy,
                special_token_ids=enlarge_token_ids,
                device=device,
                config=config,
                tokenizer=tokenizer
            )
            logger.info(f"Dynamic diffusion training completed, raw loss={loss.item():.4f}")

            # 反向传播
            loss = loss / args.gradient_accumulation_steps
            logger.info(f"Scaled loss for gradient accumulation: {loss.item():.4f}")
            loss.backward()
            logger.info("Backward pass completed")
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            epoch_samples += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                logger.info("Applying gradient clipping")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 优化器步骤
                logger.info("Optimizer step")
                optimizer.step()
                
                logger.info("Scheduler step")
                scheduler.step()
                
                logger.info("Zeroing gradients")
                optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item() * args.gradient_accumulation_steps
                logger.info(f"Completed optimization step {global_step}")

                # 日志记录
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step}/{total_steps}, Loss: {avg_loss:.4f}, LR: {lr:.2e}")
                    total_loss = 0.0

                # 保存检查点
                if global_step % args.save_steps == 0:
                    logger.info(f"Saving checkpoint at step {global_step}")
                    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, output_dir)
                    logger.info(f"Checkpoint saved to {output_dir}/checkpoint-{global_step}")

        # 每轮结束后记录
        if epoch_samples > 0:
            avg_epoch_loss = epoch_loss / epoch_samples
            logger.info(f"Epoch {epoch+1} completed, average loss: {avg_epoch_loss:.4f}")
            logger.info(f"Processed {epoch_samples} batches in this epoch")

        if global_step >= args.max_steps:
            logger.info(f"Reached max steps {args.max_steps}, early stopping")
            break

    # 保存最终模型
    logger.info("Training loop completed, saving final model")
    # save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, output_dir, is_final=True)
    logger.info("Training completed!")
    logger.info(f"Final model saved to: {output_dir}/final_model")

def main():
    """主函数"""
    args = parse_args()

    # 打印配置信息
    # logger.info("=" * 50)
    # logger.info("LLaDA Dynamic Length SFT Training")
    # logger.info("=" * 50)
    # logger.info(f"Model: {args.model_name}")
    # logger.info(f"Data: {args.data_path}")
    # logger.info(f"Output: {args.output_dir}")
    # logger.info(f"Batch size: {args.batch_size}")
    # logger.info(f"Learning rate: {args.learning_rate}")
    # logger.info(f"Max steps: {args.max_steps}")
    # logger.info(f"Max length: {args.max_length}")
    # logger.info("=" * 50)

    train_dynamic_length_sft(args)


if __name__ == "__main__":
    main()
