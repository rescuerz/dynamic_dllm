import os
import argparse
import logging
import random
from typing import Dict
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
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       help='模型数据类型（bfloat16/float16/float32）')
    parser.add_argument('--trust_remote_code', action='store_true', default=True,
                       help='是否信任远程代码（LLaDA模型需要）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    return parser.parse_args()


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
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    logger.info(f"Loading model from {args.model_name}")
    model = LLaDAModelLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
    )

    # 如果添加了新token，需要调整embedding层
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")

    return model, tokenizer


def get_enlarge_token_ids(tokenizer: AutoTokenizer) -> Dict[str, int]:
    """
    获取扩展token的ID映射

    Args:
        tokenizer: 分词器

    Returns:
        enlarge_token_ids: 扩展token的ID映射字典
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
    获取mask token ID，优先级：
    1. 模型配置中的mask_token_id
    2. tokenizer的vocab_size（词汇表大小）
    3. 默认使用词汇表大小

    Args:
        tokenizer: 分词器
        model: 模型（可选）

    Returns:
        mask_token_id: mask token的ID
    """
    # 方法1：尝试从模型配置获取
    if model is not None and hasattr(model, 'config') and hasattr(model.config, 'mask_token_id'):
        if model.config.mask_token_id is not None:
            logger.info(f"Using mask_token_id from model config: {model.config.mask_token_id}")
            return model.config.mask_token_id

    # 方法2：使用tokenizer的词汇表大小
    mask_token_id = len(tokenizer)
    logger.info(f"Using tokenizer vocab_size as mask_token_id: {mask_token_id}")
    return mask_token_id


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


def forward_diffusion_process_with_dynamic_length(input_ids, mask_token_id, eps=1e-3,
                                                 current_length=None, special_token_ids=None,
                                                 timestep=None):
    """
    支持动态长度的前向扩散过程

    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        mask_token_id: mask token的ID，用于替换被mask的位置
        eps: 最小mask概率
        current_length: 当前有效长度（支持动态长度）
        special_token_ids: 特殊token ID集合
        timestep: 指定的时间步，如果为None则随机采样

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

    # 随机mask（但保护特殊token）
    mask_indices = torch.rand((b, effective_length), device=device) < p_mask

    # 保护特殊token不被mask
    if special_token_ids is not None:
        for token_id in special_token_ids.values():
            if token_id is not None:
                special_mask = (input_ids[:, :effective_length] == token_id)
                mask_indices = mask_indices & (~special_mask)

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
    对单个样本在指定长度进行扩散训练
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

    # 计算时间步
    eps = 1e-3
    timestep = max(0.0, min(1.0, (mask_ratio - eps) / (1 - eps)))

    # 只对response部分应用扩散
    noisy_input = current_input.clone()
    if response_length > 0:
        response_part = current_input[0, prompt_len:prompt_len + response_length].unsqueeze(0)
        noisy_response, _, _, _ = forward_diffusion_process_with_dynamic_length(
            response_part, mask_token_id=mask_token_id,
            current_length=response_length, special_token_ids=special_token_ids,
            timestep=timestep
        )
        noisy_input[0, prompt_len:prompt_len + response_length] = noisy_response[0]

    # 创建p_mask
    p_mask = torch.zeros_like(noisy_input, dtype=torch.float)
    if response_length > 0:
        response_mask_prob = (1 - eps) * timestep + eps
        p_mask[0, prompt_len:prompt_len + response_length] = response_mask_prob

    # 模型前向传播
    actual_mask_indices = (noisy_input == mask_token_id)
    outputs = model(input_ids=noisy_input)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

    # 计算损失
    if actual_mask_indices.any():
        loss = loss_func(logits[actual_mask_indices], current_input[actual_mask_indices], reduction='none') / p_mask[actual_mask_indices]
        loss = loss.sum() / response_length
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def detect_sample_expansion(input_ids, prompt_length, current_length, model, special_token_ids, config, mask_token_id):
    """
    检测单个样本是否需要扩展
    """
    current_input = input_ids[:, :current_length]

    # 基于当前位置的特殊token概率进行检测
    outputs = model(input_ids=current_input)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

    # 检查最后位置的enlarge token概率
    if current_length < logits.size(1):
        position_logits = logits[0, current_length-1, :]
        probabilities = torch.softmax(position_logits, dim=-1)

        enlarge_token_id = special_token_ids.get('enlarge')
        if enlarge_token_id is not None and enlarge_token_id < logits.size(-1):
            enlarge_prob = probabilities[enlarge_token_id].item()
            if enlarge_prob > 0.7:
                return {'expand': True, 'confidence': enlarge_prob}

    return {'expand': False, 'confidence': 0.0}


def get_next_expansion_length(current_length, expansion_steps, max_length):
    """
    获取下一个扩展长度
    """
    for step in expansion_steps:
        if step > current_length and step <= max_length:
            return step
    return current_length  # 无法扩展


def create_data_collator(tokenizer):
    """创建数据整理函数"""
    def collate_fn(batch):
        """数据整理函数"""
        # 检查批次中是否有None值
        none_count = sum(1 for item in batch if item is None)
        if none_count > 0:
            logger.warning(f"Found {none_count} None items in batch of size {len(batch)}")

        # 过滤掉None值
        batch = [item for item in batch if item is not None]

        if len(batch) == 0:
            logger.warning("Empty batch after filtering None values")
            # 如果批次为空，返回空批次
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
                "prompt_lengths": torch.empty(0, dtype=torch.long),
                "enlarge_token_ids": torch.empty(0, dtype=torch.long),
            }

        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []
        prompt_lengths = []
        enlarge_token_ids = []

        for i, item in enumerate(batch):
            # 检查item是否为None或缺少必要字段
            if item is None:
                logger.error(f"Item {i} in batch is None")
                continue
            if "input_ids" not in item:
                logger.error(f"Item {i} missing 'input_ids' field")
                continue
            if item["input_ids"] is None:
                logger.error(f"Item {i} has None input_ids")
                continue

            # Padding
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)  # -100 for padding in loss calculation
            prompt_lengths.append(item["prompt_length"])
            enlarge_token_ids.append(item["enlarge_token_id"])

        # 最终检查：确保没有None值
        if any(x is None for x in input_ids):
            logger.error("Found None values in input_ids list")
            logger.error(f"input_ids: {input_ids}")
            raise ValueError("input_ids contains None values")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "enlarge_token_ids": torch.tensor(enlarge_token_ids, dtype=torch.long),
        }
    
    return collate_fn


def main():
    """主函数"""
    args = parse_args()

    # 打印配置信息
    logger.info("=" * 50)
    logger.info("LLaDA Dynamic Length SFT Training")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Max length: {args.max_length}")
    logger.info("=" * 50)

    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 使用多GPU训练
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        use_multi_gpu = True
    else:
        use_multi_gpu = False

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # 使用多GPU
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 创建配置
    config = DynamicLengthConfig(
        initial_length=128,  # 符合流程图的128 tokens起始长度
        max_length=args.max_length,
        confidence_threshold=0.7
    )

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

    # 调试：检查数据集中的第一个样本
    if len(dataset) > 0:
        first_sample = dataset[0]
        logger.info(f"First sample keys: {first_sample.keys()}")
        logger.info(f"First sample prompt: {first_sample['prompt']}")
        logger.info(f"First sample response: {first_sample['response']}")
        logger.info(f"First sample original_response: {first_sample['original_response']}")
        logger.info(f"First sample response_length: {first_sample['response_length']}")
        logger.info(f"First sample enlarge_token: {first_sample['enlarge_token']}")
        logger.info(f"First sample length_category: {first_sample['length_category']}")

    # 创建数据加载器
    collate_fn = create_data_collator(tokenizer)
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

    logger.info("Starting training...")

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            if global_step >= args.max_steps:
                break

            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 执行简化的训练步骤（这里可以集成动态长度扩散训练）
            # 为了演示，这里使用基本的语言模型训练
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 反向传播
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 优化器步骤
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item() * args.gradient_accumulation_steps

                # 日志记录
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step}/{total_steps}, Loss: {avg_loss:.4f}, LR: {lr:.2e}")
                    total_loss = 0.0

                # 保存检查点
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, output_dir)

        if global_step >= args.max_steps:
            break

    # 保存最终模型
    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, output_dir, is_final=True)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
