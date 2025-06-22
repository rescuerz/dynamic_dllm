import json
import os
import argparse
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer
from tqdm import tqdm

# 导入LLaDA模型
try:
    from model.modeling_llada import LLaDAModelLM
except ImportError:
    from llada.model.modeling_llada import LLaDAModelLM

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 设置tokenizer并行处理，避免多进程冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Special Token定义
ENLARGE_TOKENS = {
    "enlarge": "<enlarge>",  # 通用扩展token
    "enough": "<enough>"     # 结束token
}


# 长度分类阈值（针对LLaDA-8B-Instruct调整）
LENGTH_THRESHOLDS = {
    'short': (0, 256),       # 短回答
    'medium': (256, 512),   # 中等长度
    'long': (512, 1024),    # 长回答
    'very_long': (1024, 4096)  # 超长回答
}

@dataclass
class DynamicLengthConfig:
    """动态长度配置类"""
    initial_length: int = 128  # 符合流程图的128 tokens起始长度
    max_length: int = 4096
    expansion_steps: List[int] = None
    enlarge_loss_weight: float = 2.0
    confidence_threshold: float = 0.7
    expansion_check_ratio: float = 0.35  # 在30%-40%已解码token比例时检查扩展
    max_expansions: int = 5  # 最大扩展次数

    def __post_init__(self):
        if self.expansion_steps is None:
            self.expansion_steps = [128, 256, 512, 1024, 2048, 4096]  # 从128开始


class DynamicLengthDataset(Dataset):
    """动态长度数据集类"""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        config: DynamicLengthConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.enlarge_token_ids = self._get_enlarge_token_ids()
        self.examples = []

        logger.info(f"Loading data from {data_path}")
        self._load_and_process_data(data_path)
        logger.info(f"Loaded {len(self.examples)} training examples")

        # 保存处理后的数据集：example_training_data_processed.jsonl
        processed_file = data_path.replace('.jsonl', '_processed.jsonl')
        with open(processed_file, 'w', encoding='utf-8') as f:
            for example in self.examples:
                # 只需要 question 和 answer
                f.write(json.dumps({'question': example['prompt'], 'answer': example['response']}, ensure_ascii=False) + '\n')
        logger.info(f"Saved processed data to {processed_file}")

    def _get_enlarge_token_ids(self) -> Dict[str, int]:
        """获取扩展token的ID"""
        token_ids = {}
        for key, token in ENLARGE_TOKENS.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Token {token} not found in tokenizer vocabulary")
            token_ids[key] = token_id
        return token_ids

    def _load_and_process_data(self, data_path: str):
        """加载并处理训练数据"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                # 支持JSONL格式（每行一个JSON对象）
                if data_path.endswith('.jsonl'):
                    raw_data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            raw_data.append(json.loads(line))
                else:
                    # 支持标准JSON格式
                    raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            return

        for item in raw_data:
            self._process_single_sample(item)


    def _process_single_sample(self, item: Dict) -> Optional[Dict]:
        """
        处理单个训练样本，实现简化的特殊token插入策略

        策略：
        1. 精确计算token长度
        2. 根据回答长度选择特殊token（enlarge或enough）
        3. 将特殊token插入到对应的128/256/512/1024位置
        """
        try:
            # 1. 提取prompt和response的内容
            prompt, response = self._extract_conversation(item)
            if prompt is None or response is None:
                return None

            # 2. 使用tokenizer精确计算token长度
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            response_length = len(response_tokens)

            # 调试信息
            logger.debug(f"Sample: prompt_len={len(prompt_tokens)}, response_len={response_length}")

            # 3. 根据回答长度选择特殊token并插入
            if response_length <= 128:
                # 短回答：在末尾添加enough token
                enlarge_token = "enough"
                modified_response = response + ' ' + ENLARGE_TOKENS["enough"]
            else:
                # 长回答：在适当位置插入enlarge token
                enlarge_token = "enlarge"
                modified_response = self._insert_enlarge_token_at_positions(response, response_tokens, response_length)

            # 4. 返回处理后的样本
            self.examples.append({
                'prompt': prompt,                                    # 用户输入
                'response': modified_response,                       # 插入特殊token后的回答
                'original_response': response,                       # 原始回答
                'response_length': response_length,                  # token长度
                'enlarge_token': enlarge_token,                      # 使用的特殊token
                'length_category': self._get_length_category(response_length)  # 长度类别
            })

        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            return None

    def _insert_enlarge_token_at_positions(self, response: str, response_tokens: List[int], response_length: int) -> str:
        """
        在关键位置插入enlarge token

        策略：在128/256/512/1024等关键位置插入<enlarge>token
        """
        enlarge_token_text = ENLARGE_TOKENS['enlarge']

        # 定义关键位置
        key_positions = [128, 256, 512, 1024, 2048]

        # 找到需要插入的位置（小于当前长度的最大位置）
        insert_positions = [pos for pos in key_positions if pos < response_length]


        # 从后往前插入，避免位置偏移
        modified_response = response
        for pos in reversed(insert_positions):
            if pos < len(response_tokens):
                # 在指定位置插入enlarge token
                first_part = self.tokenizer.decode(response_tokens[:pos], skip_special_tokens=False)
                second_part = self.tokenizer.decode(response_tokens[pos:], skip_special_tokens=False)
                modified_response = first_part + ' ' + enlarge_token_text + ' ' + second_part

                # 更新response_tokens以便下次插入
                response_tokens = self.tokenizer.encode(modified_response, add_special_tokens=False)

        return modified_response

    def _extract_conversation(self, item: Dict) -> Tuple[str, str]:
        if 'conversations' in item:
            # ShareGPT格式：[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
            conversations = item['conversations']
            if len(conversations) >= 2:
                prompt = conversations[0].get('value', '')
                response = conversations[1].get('value', '')
                return prompt, response
        elif 'question' in item and 'answer' in item:
            # GSM8K格式：{"question": "...", "answer": "..."}
            return item['question'], item['answer']
        elif 'prompt' in item and 'response' in item:
            # 直接格式：{"prompt": "...", "response": "..."}
            return item['prompt'], item['response']
        elif 'input' in item and 'output' in item:
            # 输入输出格式：{"input": "...", "output": "..."}
            return item['input'], item['output']

        # 如果都不匹配，返回None
        return None, None



    
    def _get_length_category(self, response_length: int) -> str:
        """根据回答长度确定长度类别"""
        for category, (min_length, max_length) in LENGTH_THRESHOLDS.items():
            if response_length >= min_length and response_length <= max_length:
                return category
        return 'very_long'

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            "total_samples": len(self.examples),
            "enlarge_token_distribution": {},
            "length_distribution": {},
        }

        enlarge_counts = {}
        length_counts = {}

        for example in self.examples:
            # 统计扩展token分布
            token_key = example["enlarge_token"]
            enlarge_counts[token_key] = enlarge_counts.get(token_key, 0) + 1

            # 统计长度分布
            response_length = example["response_length"]
            length_counts[response_length] = length_counts.get(response_length, 0) + 1

        stats["enlarge_token_distribution"] = enlarge_counts
        stats["length_distribution"] = length_counts

        return stats

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        if idx >= len(self.examples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.examples)}")

        example = self.examples[idx]

        # 构建完整的输入文本（prompt + response）
        full_text = example['prompt'] + ' ' + example['response']

        # 编码为token
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors=None
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # 计算prompt长度
        prompt_encoding = self.tokenizer(
            example['prompt'],
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors=None
        )
        prompt_length = len(prompt_encoding['input_ids'])

        # 创建labels（只对response部分计算损失）
        labels = input_ids.copy()
        # 将prompt部分的labels设为-100（不计算损失）
        for i in range(prompt_length):
            if i < len(labels):
                labels[i] = -100

        # 获取enlarge token id
        enlarge_token_id = self.enlarge_token_ids[example['enlarge_token']]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'prompt_length': prompt_length,
            'enlarge_token_id': enlarge_token_id,
            'response_length': example['response_length'],
            'enlarge_token': example['enlarge_token'],
            'length_category': example['length_category'],
            'prompt': example['prompt'],
            'response': example['response'],
            'original_response': example['original_response']
        }


class LLaDADiffusionTrainer:
    """LLaDA扩散模型动态长度训练器"""

    def __init__(self, model: LLaDAModelLM, tokenizer: AutoTokenizer, config: DynamicLengthConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.mask_token_id = len(tokenizer)  # 使用词汇表大小作为mask token ID

        # 获取扩展token的ID
        self.enlarge_token_ids = {}
        for key, token in ENLARGE_TOKENS.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                self.enlarge_token_ids[key] = token_id

        logger.info(f"Initialized LLaDA trainer with mask_token_id={self.mask_token_id}")
        logger.info(f"Enlarge token IDs: {self.enlarge_token_ids}")

    def add_noise(self, input_ids: torch.Tensor, prompt_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LLaDA风格的加噪过程

        Args:
            input_ids: [batch_size, seq_len] 输入token序列
            prompt_lengths: [batch_size] 每个样本的prompt长度

        Returns:
            noisy_input: 加噪后的输入
            mask_ratio: 实际的mask比例
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 随机采样mask比例，对每个样本独立采样
        mask_ratios = torch.rand(batch_size, device=device)

        noisy_input = input_ids.clone()

        for i in range(batch_size):
            prompt_len = prompt_lengths[i].item()
            response_len = seq_len - prompt_len

            if response_len <= 0:
                continue

            # 计算需要mask的token数量
            num_mask = int(response_len * mask_ratios[i].item())

            if num_mask > 0:
                # 在response部分随机选择位置进行mask
                response_indices = torch.arange(prompt_len, seq_len, device=device)
                mask_indices = response_indices[torch.randperm(response_len, device=device)[:num_mask]]
                noisy_input[i, mask_indices] = self.mask_token_id

        return noisy_input, mask_ratios

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                    mask_positions: torch.Tensor, enlarge_token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        计算扩散模型的损失

        Args:
            logits: [batch_size, seq_len, vocab_size] 模型输出
            targets: [batch_size, seq_len] 目标token
            mask_positions: [batch_size, seq_len] mask位置的布尔张量
            enlarge_token_positions: [batch_size, seq_len] 扩展token位置的布尔张量

        Returns:
            loss: 计算得到的损失
        """
        # 只对mask位置计算损失
        if mask_positions.any():
            mask_logits = logits[mask_positions]
            mask_targets = targets[mask_positions]

            # 基础交叉熵损失
            loss = F.cross_entropy(mask_logits, mask_targets, reduction='mean')

            # 如果有扩展token位置，给予更高权重
            if enlarge_token_positions is not None and enlarge_token_positions.any():
                enlarge_mask_positions = mask_positions & enlarge_token_positions
                if enlarge_mask_positions.any():
                    enlarge_logits = logits[enlarge_mask_positions]
                    enlarge_targets = targets[enlarge_mask_positions]
                    enlarge_loss = F.cross_entropy(enlarge_logits, enlarge_targets, reduction='mean')

                    # 加权组合损失
                    loss = 0.7 * loss + 0.3 * self.config.enlarge_loss_weight * enlarge_loss
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss

    def detect_expansion_need(self, logits: torch.Tensor, current_step: int, total_steps: int) -> List[Dict]:
        """
        检测是否需要扩展长度

        Args:
            logits: [batch_size, seq_len, vocab_size] 模型输出
            current_step: 当前训练步数
            total_steps: 总训练步数

        Returns:
            expansion_decisions: 每个样本的扩展决策
        """
        batch_size = logits.size(0)
        progress = current_step / total_steps

        # 只在特定进度区间检测扩展
        if not (0.4 <= progress <= 0.6):
            return [{"expand": False, "target_length": None} for _ in range(batch_size)]

        decisions = []
        for i in range(batch_size):
            # 检查最后几个位置的token概率
            last_logits = logits[i, -3:, :]  # 检查最后3个位置
            probs = F.softmax(last_logits, dim=-1)

            max_prob = 0.0
            target_length = None

            # 检查扩展token的概率
            for token_key, token_id in self.enlarge_token_ids.items():
                if token_key != "enough" and token_id < logits.size(-1):
                    token_probs = probs[:, token_id]
                    max_token_prob = token_probs.max().item()

                    if max_token_prob > max_prob and max_token_prob > self.config.confidence_threshold:
                        max_prob = max_token_prob
                        if token_key == "enlarge_512":
                            target_length = 512
                        elif token_key == "enlarge_1024":
                            target_length = 1024
                        elif token_key == "enlarge_2048":
                            target_length = 2048

            decisions.append({
                "expand": target_length is not None,
                "target_length": target_length,
                "confidence": max_prob
            })

        return decisions





def forward_diffusion_process_with_dynamic_length(input_ids, total_dim=32000, eps=1e-3,
                                                 current_length=None, special_token_ids=None,
                                                 timestep=None):
    """
    支持动态长度的前向扩散过程

    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        total_dim: 词汇表大小，用作mask token ID
        eps: 最小mask概率
        current_length: 当前有效长度（支持动态长度）
        special_token_ids: 特殊token ID集合
        timestep: 指定的时间步，如果为None则随机采样

    Returns:
        tuple: (noisy_input, p_mask, mask_indices, actual_timestep)
    """
    b, l = input_ids.shape
    device = input_ids.device

    # 动态长度处理
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
        total_dim,  # 使用total_dim作为mask token
        input_ids[:, :effective_length]
    )

    return noisy_input, full_p_mask, mask_indices, t


def detect_length_expansion_by_decoded_ratio(logits, mask_positions, special_token_ids, current_length=128, total_length=None, detection_ratio=(0.3, 0.4)):
    """
    基于已解码token比例检测长度扩展需求

    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        mask_positions: 当前mask位置 [batch_size, seq_len] 布尔张量
        special_token_ids: 特殊token ID映射
        current_length: 当前序列长度
        total_length: 总序列长度（如果为None，使用logits.size(1)）
        detection_ratio: 检测比例区间，默认(0.3, 0.4)

    Returns:
        list: 每个样本的扩展决策
    """
    batch_size = logits.size(0)
    if total_length is None:
        total_length = current_length

    expansion_decisions = []
    detection_start, detection_end = detection_ratio

    for i in range(batch_size):
        # 计算已解码token数（非mask的token数）
        if current_length > 0:
            current_mask = mask_positions[i, :current_length]
            decoded_tokens = current_length - current_mask.sum().item()
            decoded_ratio = decoded_tokens / current_length
        else:
            decoded_ratio = 0.0

        # 检查已解码比例是否在检测窗口内（30%-40%）
        if not (detection_start <= decoded_ratio <= detection_end):
            expansion_decisions.append({
                'expand': False,
                'decoded_ratio': decoded_ratio
            })
            continue

        # 在当前长度边界检查是否有enlarge token
        should_expand = False
        max_prob = 0.0

        if current_length < logits.size(1):
            position_logits = logits[i, current_length-1, :]
            probabilities = torch.softmax(position_logits, dim=-1)

            # 只检查enlarge token的概率
            enlarge_token_id = special_token_ids.get('enlarge')
            if enlarge_token_id is not None and enlarge_token_id < logits.size(-1):
                enlarge_prob = probabilities[enlarge_token_id].item()
                if enlarge_prob > 0.7:  # 阈值
                    should_expand = True
                    max_prob = enlarge_prob

        expansion_decisions.append({
            'expand': should_expand,
            'confidence': max_prob,
            'decoded_ratio': decoded_ratio,
            'decoded_tokens': decoded_tokens,
            'current_length': current_length
        })

    return expansion_decisions


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
    special_tokens = list(ENLARGE_TOKENS.values())
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


def train_dynamic_length_sft(args):
    """主训练函数"""
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
        enlarge_loss_weight=2.0,
        confidence_threshold=0.7
    )

    # 创建训练器
    trainer = LLaDADiffusionTrainer(model, tokenizer, config)

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

    for _ in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            if global_step >= args.max_steps:
                break

            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 执行多次扩展的动态长度扩散训练步骤
            loss = train_dynamic_diffusion_step_multi_expansion(
                input_ids=batch["input_ids"],
                prompt_length=batch["prompt_lengths"],
                model=model,
                loss_func=torch.nn.functional.cross_entropy,
                special_token_ids=trainer.enlarge_token_ids,
                device=device,
                config=config
            )

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
                # if global_step % args.save_steps == 0:
                #     save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, output_dir)

        if global_step >= args.max_steps:
            break

    # 保存最终模型
    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, output_dir, is_final=True)
    logger.info("Training completed!")





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

    train_dynamic_length_sft(args)


def train_dynamic_diffusion_step_multi_expansion(input_ids, prompt_length, model, loss_func,
                                               special_token_ids, device, config):
    """
    支持多次动态扩展的扩散训练

    新的逻辑：
    1. 从128 tokens开始训练
    2. 基于已解码token比例（30%-40%）检测扩展需求
    3. 支持多次扩展：128→256→512→1024→2048→4096
    4. 每次扩展都对完整序列重新进行扩散训练
    """
    total_dim = 32000  # 词汇表大小
    initial_length = config.initial_length  # 从配置中获取初始长度
    max_expansions = config.max_expansions  # 从配置中获取最大扩展次数

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    expansion_count = 0
    current_length = initial_length

    # 多次扩展循环
    for expansion_round in range(max_expansions):
        # 确保当前长度不超过输入序列长度
        if current_length >= input_ids.size(1):
            break

        # 1. 当前长度的扩散训练
        current_input = input_ids[:, :current_length]

        # 应用扩散过程
        noisy_input, p_mask, _, _ = forward_diffusion_process_with_dynamic_length(
            current_input, total_dim=total_dim, current_length=current_length,
            special_token_ids=special_token_ids
        )

        # 创建位置索引，保护prompt部分
        temp_tensor = torch.arange(noisy_input.size(1), device=device).expand(noisy_input.size(0), noisy_input.size(1))
        prompt_index = (temp_tensor < prompt_length.unsqueeze(1))

        # 保持prompt部分不被mask
        noisy_input[prompt_index] = current_input[prompt_index].clone()

        # 记录实际被mask的位置
        actual_mask_indices = (noisy_input == total_dim)

        # 模型前向传播
        outputs = model(input_ids=noisy_input)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # 计算当前长度的损失
        if actual_mask_indices.any():
            current_loss = loss_func(logits[actual_mask_indices], current_input[actual_mask_indices], reduction='none') / p_mask[actual_mask_indices]
            current_loss = current_loss.sum() / (current_input.shape[0] * current_length - prompt_length.sum())
        else:
            current_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 累加损失
        total_loss = total_loss + current_loss

        # 2. 检测是否需要扩展（基于已解码token比例）
        detection_ratio = (config.expansion_check_ratio - 0.05, config.expansion_check_ratio + 0.05)  # 基于config计算检测区间
        expansion_decisions = detect_length_expansion_by_decoded_ratio(
            logits, mask_positions=actual_mask_indices,
            special_token_ids=special_token_ids, current_length=current_length,
            detection_ratio=detection_ratio
        )

        # 3. 处理扩展决策
        should_expand = False

        # 检查是否有任何样本需要扩展
        for decision in expansion_decisions:
            if decision['expand']:
                should_expand = True
                expansion_count += 1
                break

        # 如果需要扩展，按照预定义步骤扩展到下一个长度
        if should_expand:
            # 找到下一个扩展长度
            expansion_steps = config.expansion_steps
            next_length = current_length
            for step in expansion_steps:
                if step > current_length and step <= input_ids.size(1):
                    next_length = step
                    break

            if next_length > current_length:
                current_length = next_length
                logger.debug(f"Expansion round {expansion_round + 1}: extending to {current_length} tokens")
            else:
                # 没有更大的扩展步骤，结束循环
                break
        else:
            # 没有扩展需求，结束循环
            break

    # 4. 返回平均损失
    if expansion_count > 0:
        # 如果有扩展，对损失进行平均
        total_loss = total_loss / (expansion_round + 1)

    return total_loss





if __name__ == "__main__":
    main()


# 使用示例：
# python llada/finetune_dynamic_length.py \
#     --model_name GSAI-ML/LLaDA-8B-Instruct \
#     --data_path llada/example_training_data.jsonl \
#     --output_dir ./output/llada_dynamic \
#     --batch_size 1 \
#     --learning_rate 5e-6 \
#     --max_steps 1000 \
#     --gradient_accumulation_steps 8 \
#     --save_steps 500 \
#     --logging_steps 100 \
#     --max_length 2048
#
# 特殊token说明：
# - <enlarge>: 通用扩展token，表示需要扩展长度
# - <enough>: 结束token，表示当前长度已足够
#
# 多次扩展流程：128→256→512→1024→2048→4096
