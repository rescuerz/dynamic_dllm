import json
import tempfile
import os
from transformers import AutoTokenizer
from finetune_dynamic_length import DynamicLengthDataset, DynamicLengthConfig, ENLARGE_TOKENS

test_list = [1, 3, 5, 7, 9]

def test_dataset_creation():
    """测试数据集创建"""
    print("1. 测试数据集创建...")
    
    # 使用GSM8K数据集
    data_file = "dataset/gsm8k_train.jsonl"
    print(f"使用GSM8K数据集: {data_file}")


    try:
        # 初始化tokenizer
        print("2. 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 添加特殊tokens
        special_tokens = list(ENLARGE_TOKENS.values())
        existing_tokens = set(tokenizer.get_vocab().keys())
        new_tokens = [token for token in special_tokens if token not in existing_tokens]
        
        if new_tokens:
            print(f"3. 添加新的特殊tokens: {new_tokens}")
            tokenizer.add_tokens(new_tokens)
        
        # 创建配置
        config = DynamicLengthConfig(
            initial_length=128,
            max_length=2048,
            max_expansions=5
        )
        
        # 创建数据集
        print("4. 创建数据集...")
        dataset = DynamicLengthDataset(
            data_path=data_file,
            tokenizer=tokenizer,
            config=config
        )

        print(f"✅ 数据集创建成功！共 {len(dataset)} 个样本")
        return dataset, tokenizer

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        raise

def test_special_token_insertion(dataset, tokenizer):
    """测试特殊token插入逻辑"""
    print("\n5. 测试特殊token插入逻辑...")
    
    for i, example in enumerate(dataset.examples):
        if i in test_list:
            print(f"\n--- 样本 {i+1} ---")
            print(f"原始回答长度: {example['response_length']} tokens")
            print(f"使用的特殊token: {example['enlarge_token']}")
            
            # 检查原始回答
            original_tokens = tokenizer.encode(example['original_response'], add_special_tokens=False)
            print(f"原始回答实际token数: {len(original_tokens)}")
            
            # 检查修改后的回答
            modified_tokens = tokenizer.encode(example['response'], add_special_tokens=False)
            print(f"修改后回答token数: {len(modified_tokens)}")
            
            # 检查特殊token是否正确插入
            enlarge_token_id = tokenizer.convert_tokens_to_ids(ENLARGE_TOKENS[example['enlarge_token']])
            if enlarge_token_id in modified_tokens:
                positions = [j for j, token_id in enumerate(modified_tokens) if token_id == enlarge_token_id]
                print(f"特殊token位置: {positions}")
            else:
                print("特殊token未找到")
            
            # 显示部分内容
            print(f"原始回答: {example['original_response']}")
            print(f"修改后回答: {example['response']}")
            
            # 验证逻辑
            if example['response_length'] <= 128:
                assert example['enlarge_token'] == 'enough', f"短回答应该使用enough token，但使用了{example['enlarge_token']}"
                assert example['response'].endswith(ENLARGE_TOKENS['enough']), "短回答应该在末尾添加enough token"
            else:
                assert example['enlarge_token'] == 'enlarge', f"长回答应该使用enlarge token，但使用了{example['enlarge_token']}"
                assert ENLARGE_TOKENS['enlarge'] in example['response'], "长回答应该包含enlarge token"

def test_dataset_getitem(dataset, tokenizer):
    """测试数据集的__getitem__方法"""
    print("\n6. 测试数据集__getitem__方法...")
    
    for i in test_list:  # 测试前3个样本
        print(f"\n--- 测试样本 {i+1} ---")
        
        item = dataset[i]
        
        # 检查返回的字段
        expected_keys = [
            'input_ids', 'attention_mask', 'labels', 'prompt_length',
            'enlarge_token_id', 'response_length', 'enlarge_token',
            'length_category', 'prompt', 'response', 'original_response'
        ]
        
        for key in expected_keys:
            assert key in item, f"缺少字段: {key}"
        
        print(f"input_ids长度: {len(item['input_ids'])}")
        print(f"prompt长度: {item['prompt_length']}")
        print(f"response长度: {item['response_length']}")
        print(f"enlarge_token_id: {item['enlarge_token_id']}")
        print(f"enlarge_token: {item['enlarge_token']}")
        
        # 验证labels的设置（prompt部分应该是-100）
        labels = item['labels']
        prompt_length = item['prompt_length']
        
        # 检查prompt部分的labels是否为-100
        prompt_labels = labels[:prompt_length]
        assert all(label == -100 for label in prompt_labels), "Prompt部分的labels应该全部为-100"
        
        # 检查response部分是否有有效的labels
        response_labels = labels[prompt_length:]
        valid_labels = [label for label in response_labels if label != -100]
        print(f"有效labels数量: {len(valid_labels)}")
        
        print("✅ 样本验证通过")


def main():
    """主测试函数"""
    print("🚀 开始测试DynamicLengthDataset...")
    
    try:
        # 测试数据集创建
        dataset, tokenizer = test_dataset_creation()
        
        # 测试特殊token插入
        test_special_token_insertion(dataset, tokenizer)
        
        # 测试__getitem__方法
        test_dataset_getitem(dataset, tokenizer)
        
        print("\n🎉 所有测试通过！DynamicLengthDataset工作正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
