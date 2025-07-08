<!-- @format -->

# LLaDA 动态长度监督微调 (Dynamic Length SFT)

这是一个基于 LLaDA 模型的动态长度监督微调项目。该项目实现了一种创新的训练方法，通过扩散过程和特殊 token 来动态控制生成文本的长度。


## 目录结构
我是在fast-dllm代码的基础上修改，针对llada的动态生成长度微调，添加代码文件dynamic_length_sft.py, dynamic_length_dataset.py

```
dynamic_dllm/
├── llada/                          # LLaDA相关代码
│   ├── model/                      # 模型定义
│   │   ├── modeling_llada.py       # LLaDA模型实现
│   │   ├── configuration_llada.py  # 模型配置
│   │   └── __init__.py
│   ├── dynamic_length_sft.py       # 动态长度SFT训练脚本
│   ├── dynamic_length_dataset.py   # 动态长度数据集处理
│   ├── generate.py                 # 文本生成脚本
│   ├── chat.py                     # 交互式对话脚本
│   ├── eval_llada.py              # 模型评估脚本
│   ├── run.sh                      # 训练启动脚本
│   └── example_training_data.jsonl # 示例训练数据
├── dream/                          # DREAM相关代码
├── dataset/                        # 数据集目录
│   ├── gsm8k_train.jsonl          # GSM8K训练数据
│   └── gsm8k_test.jsonl           # GSM8K测试数据
├── requirements.txt                # 依赖包列表
└── README.md                       # 项目说明文档
```


## 扩散训练过程

### 1. 数据预处理

`DynamicLengthDataset` 类负责数据预处理：

**特殊 token 插入**：根据回答长度在特定位置插入特殊 token
   - 短回答(≤64 tokens)：末尾添加 `<enough>`
   - 长回答(>64 tokens)：在 64、128、256、512、1024 位置插入 `<enlarge>`


### 2. 扩散训练

`train_dynamic_diffusion_step_multi_expansion` 函数实现核心训练逻辑：

1. **样本级独立训练**：每个样本独立进行多轮扩展训练
2. **前向扩散**：随机 mask 部分 token，模拟噪声添加过程
3. **扩展决策**：
   - 检查未被 mask 的特殊 token（直接决策）
   - 使用模型预测特殊 token 概率（间接决策）
4. **损失计算**：只对被 mask 的位置计算交叉熵损失

### 3. 扩展决策机制

系统使用两种方式进行扩展决策：

**直接决策**：

- 如果在关键位置发现未被 mask 的 `<enlarge>` token → 扩展
- 如果在关键位置发现未被 mask 的 `<enough>` token → 停止

**模型预测**：

- 计算模型在关键位置预测特殊 token 的概率
- 如果概率超过阈值(默认 0.7) → 根据 token 类型决策



