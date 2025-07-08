import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# 设置日志
logger = logging.getLogger(__name__)

# Special Token定义
SPECIAL_TOKENS = {
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
    initial_length: int = 64  # 符合流程图的64 tokens起始长度
    max_length: int = 4096
    expansion_steps: List[int] = None
    confidence_threshold: float = 0.7
    expansion_check_ratio: float = 0.35  # 在30%-40%已解码token比例时检查扩展
    max_expansions: int = 5  # 最大扩展次数
    exclude_special_tokens_from_attention: bool = False  # 是否将special token排除在注意力机制之外

    def __post_init__(self):
        if self.expansion_steps is None:
            self.expansion_steps = [64, 128, 256, 512, 1024, 2048, 4096]  # 从64开始


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
        self._log_special_token_exclusion = False  # 控制是否记录special token排除日志

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
        for key, token in SPECIAL_TOKENS.items():
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
        3. 将特殊token插入到对应的64/128/256/512/1024位置
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
            if response_length <= 64:
                # 短回答：在末尾添加enough token
                enlarge_token = "enough"
                modified_response = response + ' ' + SPECIAL_TOKENS["enough"]
            else:
                # 长回答：只在适当位置插入enlarge token，末尾不添加enough token
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

        目标：确保<enlarge> token最终位于下标63、127、255、511、1023等位置
        策略：从前往后插入，每次插入时计算正确的插入位置，确保最终位置准确
        直接在token级别操作，避免空格导致的位置偏移
        """
        # 获取enlarge token的ID
        enlarge_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['enlarge'])

        # 定义目标最终位置（0基索引）
        target_final_positions = [63, 127, 255, 511, 1023, 2047]  # 对应64、128、256、512、1024、2048位置

        # 复制原始token列表
        modified_tokens = response_tokens.copy()

        # 从前往后插入，确保每个<enlarge> token最终位于正确位置
        for final_pos in target_final_positions:
            # 检查原始长度是否足够长，需要插入这个位置的token
            if final_pos < response_length:
                # 直接在目标最终位置插入
                # 由于我们从前往后插入，当前的final_pos就是我们要插入的位置
                insert_pos = final_pos

                # 确保插入位置在有效范围内
                if insert_pos >= 0 and insert_pos < len(modified_tokens):
                    # 在计算出的位置插入enlarge token ID
                    modified_tokens.insert(insert_pos, enlarge_token_id)

                    logger.debug(f"Inserted <enlarge> at index {insert_pos}, final position will be {final_pos}")

        # 将修改后的token列表解码为文本
        modified_response = self.tokenizer.decode(modified_tokens, skip_special_tokens=False)

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

    def enable_special_token_logging(self, enable: bool = True):
        """启用或禁用special token排除的日志记录"""
        self._log_special_token_exclusion = enable

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
        attention_mask = encoding['attention_mask'].copy()

        # 计算prompt长度
        prompt_encoding = self.tokenizer(
            example['prompt'],
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors=None
        )
        prompt_length = len(prompt_encoding['input_ids'])

        # 修改attention_mask：根据配置决定是否将special token排除在注意力机制之外
        if self.config.exclude_special_tokens_from_attention:
            special_token_count = 0
            for i, token_id in enumerate(input_ids):
                if token_id in self.enlarge_token_ids.values():
                    attention_mask[i] = 0  # special token不参与注意力计算
                    special_token_count += 1

            if special_token_count > 0:
                # 只在有special token时记录日志，避免过多输出
                if hasattr(self, '_log_special_token_exclusion') and self._log_special_token_exclusion:
                    print(f"Dataset: Excluded {special_token_count} special tokens from attention mask")

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


def test_dataset_processing():
    """测试数据集处理功能"""
    import os
    from transformers import AutoTokenizer

    print("=" * 60)
    print("测试 GSM8K 数据集处理")
    print("=" * 60)

    # 检查数据文件是否存在
    # 从llada目录向上查找dataset目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset_dir = os.path.join(parent_dir, "dataset")

    gsm8k_files = [
        "gsm8k_train.jsonl",
        "gsm8k_test.jsonl"
    ]

    available_files = []
    for file in gsm8k_files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            available_files.append(file_path)
            print(f"✓ 找到数据文件: {file_path}")
        else:
            print(f"✗ 未找到数据文件: {file_path}")

    if not available_files:
        print("错误：未找到任何GSM8K数据文件")
        return

    # 使用第一个可用文件进行测试
    test_file = available_files[0]
    print(f"\n使用文件进行测试: {test_file}")

    try:
        # 初始化tokenizer（使用一个轻量级的tokenizer进行测试）
        print("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",  # 使用较小的模型进行测试
            padding_side="right",
            use_fast=True
        )

        # 添加特殊token
        special_tokens = list(SPECIAL_TOKENS.values())
        existing_tokens = set(tokenizer.get_vocab().keys())
        new_tokens = [token for token in special_tokens if token not in existing_tokens]

        if new_tokens:
            special_tokens_dict = {'additional_special_tokens': new_tokens}
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"添加了 {num_added} 个特殊token: {new_tokens}")

        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"pad token: {tokenizer.pad_token}")
        print(f"eos token: {tokenizer.eos_token}")

        # 创建配置
        config = DynamicLengthConfig()

        print(f"配置信息:")
        print(f"  初始长度: {config.initial_length}")
        print(f"  最大长度: {config.max_length}")
        print(f"  扩展步骤: {config.expansion_steps}")

        # 创建数据集
        print(f"\n正在处理数据集: {test_file}")
        dataset = DynamicLengthDataset(
            data_path=test_file,
            tokenizer=tokenizer,
            config=config
        )

        # 打印统计信息
        stats = dataset.get_statistics()
        print(f"\n数据集统计信息:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  扩展token分布: {stats['enlarge_token_distribution']}")


        # 获取特殊token的ID
        enlarge_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["enlarge"])
        enough_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["enough"])
        special_token_ids = [enlarge_token_id, enough_token_id]

        print(f"\n特殊Token ID映射:")
        print(f"  {SPECIAL_TOKENS['enlarge']} -> ID: {enlarge_token_id}")
        print(f"  {SPECIAL_TOKENS['enough']} -> ID: {enough_token_id}")

        # 显示几个样本示例
        print(f"\n样本示例:")
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            response = sample['response']
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            response_length = len(response_tokens)

            # 寻找special token的位置
            enlarge_positions = [idx for idx, token_id in enumerate(response_tokens) if token_id == enlarge_token_id]
            enough_positions = [idx for idx, token_id in enumerate(response_tokens) if token_id == enough_token_id]

            print(f"\n样本 {i+1}:")
            print(f"  Prompt: {sample['prompt']}")
            print(f"  Response: {sample['response']}")
            print(f"  Original Response: {sample['original_response']}")
            print(f"  Response长度: {sample['response_length']} tokens")
            print(f"  扩展Token: {sample['enlarge_token']}")
            print(f"  长度类别: {sample['length_category']}")
            print(f"  Input IDs长度: {len(sample['input_ids'])}")
            print(f"  Response token数量: {len(response_tokens)}")
            print(f"  <enlarge> token位置: {enlarge_positions}")
            print(f"  <enough> token位置: {enough_positions}")

            # 验证特殊token是否在预期位置
            if sample['enlarge_token'] == 'enlarge' and enlarge_positions:
                original_length = sample['response_length']
                target_final_positions = [63, 127, 255, 511, 1023, 2047]  # 期望的最终位置（0基索引）

                # 找到应该插入的最终位置
                expected_final_positions = [pos for pos in target_final_positions if pos < original_length]

                print(f"  原始长度: {original_length}")
                print(f"  预期<enlarge>最终位置: {expected_final_positions}")
                print(f"  实际<enlarge>位置: {enlarge_positions}")

                if enlarge_positions == expected_final_positions:
                    print(f"  ✅ 位置完全匹配!")
                else:
                    print(f"  ⚠️  位置不匹配")
            elif sample['enlarge_token'] == 'enough' and enough_positions:
                print(f"  ✅ <enough> token在末尾位置")

        print(f"\n✓ 数据集处理测试完成！")
        print(f"处理后的数据已保存到: {test_file.replace('.jsonl', '_processed.jsonl')}")

    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    """运行测试"""
    print("动态长度数据集处理测试")
    print("选择测试模式:")
    print("1. 数据特征分析")
    print("2. 数据集处理测试")
    print("3. 运行所有测试")

    try:
        choice = input("请输入选择 (1/2/3): ").strip()

        if choice == "1":
            # analyze_gsm8k_data()
            pass
        elif choice == "2":
            test_dataset_processing()
        elif choice == "3":
            # analyze_gsm8k_data()
            print("\n" + "="*60 + "\n")
            test_dataset_processing()
        else:
            print("无效选择，运行所有测试...")
            # analyze_gsm8k_data()
            print("\n" + "="*60 + "\n")
            test_dataset_processing()

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
