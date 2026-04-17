# KernelGYM SFT 训练代码走读与框架介绍

## 一、项目概述

KernelGYM 是一个面向 GPU Kernel 生成任务的训练与评估环境，支持 **SFT（Supervised Fine-Tuning）** 和 **RL（Reinforcement Learning）** 两种训练范式。本文档聚焦于 SFT 训练部分的代码走读与框架解析。

### 核心定位
- **SFT 阶段**：使用高质量的 kernel 生成数据对基座模型进行监督微调（cold start）
- **RL 阶段**：在 SFT 模型基础上，通过强化学习进一步优化 kernel 生成的正确性和性能
- **底层框架**：基于 VERL（Versatile Reinforcement Learning）框架扩展

---

## 二、启动脚本解析

### 2.1 脚本路径
```
drkernel/kernel/scripts/sft/8b-coldstart.sh
```

### 2.2 脚本执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     启动脚本执行流程                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 设置默认参数（batch size、max length、learning rate 等）        │
│  2. 检测分布式环境（节点数、GPU数、主节点地址等）                     │
│  3. 检测设备类型（NPU 或 CUDA）                                    │
│  4. 解析命令行参数（支持动态覆盖默认值）                            │
│  5. 生成唯一的 RUN_NAME（基于参数组合）                            │
│  6. 初始化日志系统                                                 │
│  7. 通过 torchrun 启动分布式训练                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 关键默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_BATCH_SIZE` | 64 | 全局训练 batch size |
| `MICRO_BATCH_SIZE_PER_GPU` | 2 | 每张卡的 micro batch size |
| `MAX_LENGTH` | 18432 | 序列最大长度 |
| `TOTAL_EPOCHS` | 4 | 训练轮数 |
| `SAVE_FREQ` | 50 | 每 50 step 保存一次 checkpoint |
| `MODEL_NAME` | qwen3-8b-base | 基座模型 |
| `DATASET_NAME` | hkust-nlp/drkernel-coldstart-8k | 训练数据集 |
| `LEARNING_RATE` | 2e-5 | 学习率 |
| `SP_SIZE` | 4 | Ulysses 序列并行大小 |
| `TRUNCATION` | right | 超长序列截断策略 |

### 2.4 分布式环境配置

脚本自动检测并配置以下环境变量：
- `NNODES`：节点数量（从 `ARNOLD_WORKER_NUM` 获取）
- `GPUS_PER_NODE`：每节点 GPU 数量（从 `ARNOLD_WORKER_GPU` 获取，默认 8）
- `NODE_RANK`：当前节点 rank
- `MASTER_ADDR`：主节点地址
- `MASTER_PORT`：主节点端口（自动检测可用端口）

### 2.5 训练启动命令

```bash
torchrun --nproc-per-node $GPUS_PER_NODE \
  --master-addr $MASTER_ADDR \
  --node-rank $NODE_RANK \
  --master-port $MASTER_PORT \
  --nnodes $NNODES \
  -m kernel.fsdp_sft_trainer \
  data.multiturn.enable=True \
  data.train_files=$ACTUAL_DATA_PATH \
  ...
```

核心入口模块：`kernel.fsdp_sft_trainer`

---

## 三、SFT 训练框架架构

### 3.1 整体架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                        SFT 训练架构                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  启动脚本     │───▶│  fsdp_sft_trainer│───▶│  FSDPSFTTrainer  │  │
│  │  (shell)     │    │  (入口模块)       │    │  (训练器类)       │  │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                       │             │
│                      ┌────────────────────────────────┼──────────┐  │
│                      │                                │          │  │
│              ┌───────▼───────┐              ┌─────────▼───────┐  │  │
│              │  SFTDataset   │              │  FSDP Model     │  │  │
│              │  (数据集)      │              │  (分布式模型)    │  │  │
│              └───────────────┘              └─────────────────┘  │  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    分布式策略                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │   FSDP      │  │  Ulysses SP │  │  CPU Offload        │  │  │
│  │  │  (全分片)    │  │  (序列并行)  │  │  (可选)             │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心模块说明

| 模块 | 路径 | 职责 |
|------|------|------|
| 启动脚本 | `scripts/sft/8b-coldstart.sh` | 参数配置、环境检测、torchrun 启动 |
| 入口模块 | `kernel/fsdp_sft_trainer.py` | 进程组初始化、DeviceMesh 创建、资源协调 |
| 训练器 | `verl_patch/trainer/code/fsdp_sft_trainer.py` | 模型加载、FSDP 包装、训练循环、checkpoint |
| 数据集 | `verl_patch/utils/dataset/sft_dataset.py` | 数据加载、tokenization、loss mask 构建 |
| 配置 | `verl_patch/trainer/code/config/sft_trainer.yaml` | 默认配置参数 |

---

## 四、核心代码走读

### 4.1 入口模块：`kernel/fsdp_sft_trainer.py`

#### 4.1.1 整体流程

```python
@hydra.main(config_path="../verl_patch/trainer/code/config", config_name="sft_trainer")
def main(config):
    # 1. 初始化分布式进程组
    local_rank, rank, world_size = initialize_global_process_group()
    
    # 2. 初始化日志系统（双通道：控制台 + 文件）
    logger = setup_logger(rank, log_dir)
    
    # 3. 创建 FSDP DeviceMesh
    device_mesh = init_device_mesh(device_type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    
    # 4. 创建 Ulysses 序列并行 DeviceMesh
    ulysses_device_mesh = init_device_mesh(device_type, mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp"))
    
    # 5. 拷贝模型到本地并加载 Tokenizer
    local_model_path = copy_to_local(src=config.model.partial_pretrain)
    tokenizer = hf_tokenizer(local_model_path)
    
    # 6. Qwen3 模型特殊处理（修复 chat_template）
    if is_qwen3 and not coder:
        tokenizer.chat_template = QWEN3CHATTEMPLATE
    
    # 7. 构建训练集和验证集
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
    
    # 8. 初始化 Trainer
    trainer = FSDPSFTTrainer(config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)
    
    # 9. 启动训练
    trainer.fit()
```

#### 4.1.2 日志系统

采用双通道日志设计：
- **控制台**：INFO 级别，带 rank 标识，方便实时监控
- **文件**：DEBUG 级别，包含文件名/行号，支持 RotatingFileHandler（单文件 100MB，保留 5 个备份）
- **Master 日志**：Rank 0 额外写入全局汇总日志 `sft_master.log`

### 4.2 训练器类：`FSDPSFTTrainer`

#### 4.2.1 初始化流程 (`__init__`)

```
┌─────────────────────────────────────────────────────────────┐
│                    FSDPSFTTrainer 初始化                      │
├─────────────────────────────────────────────────────────────┤
│  1. _normalize_config_bsz()                                 │
│     - 根据 DP size 归一化 batch size                         │
│     - 验证 batch size 整除关系                                │
│                                                              │
│  2. _build_dataloader(train_dataset, val_dataset)           │
│     - 创建 DistributedSampler                                │
│     - 构建 DataLoader（num_workers=8, pin_memory=True）       │
│                                                              │
│  3. _build_model_optimizer()                                │
│     - 加载预训练模型（AutoModelForCausalLM）                  │
│     - 应用 Monkey Patch（序列并行支持）                       │
│     - 可选 LoRA 适配                                         │
│     - Gradient Checkpointing                                 │
│     - FSDP 包装（FULL_SHARD 策略）                           │
│     - 创建 AdamW 优化器                                      │
│     - 创建 Cosine/WSD 学习率调度器                           │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2.2 FSDP 配置

```python
mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,    # 参数精度
    reduce_dtype=torch.float32,    # 梯度归约精度
    buffer_dtype=torch.float32     # 缓冲区精度
)

fsdp_model = FSDP(
    module=model,
    auto_wrap_policy=get_fsdp_wrap_policy(...),  # 自动包装策略
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 全分片
    mixed_precision=mixed_precision,
    device_mesh=device_mesh,
    sync_module_states=True,       # 同步模块状态
    cpu_offload=cpu_offload,       # 可选 CPU offload
    use_orig_params=False,         # 使用扁平参数
)
```

#### 4.2.3 训练循环 (`fit`)

```python
def fit(self):
    # 初始化 tracking（WandB / Console）
    tracking = Tracking(project_name, experiment_name, logger)
    
    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)
        
        for data in tqdm(train_dataloader):
            global_step += 1
            data = TensorDict(data).to(device)
            
            # 训练 step
            metric = self.training_step(data)
            
            # 记录指标
            tracking.log(data=metric, step=global_step)
            
            # 定期保存 checkpoint
            if global_step % save_freq == 0:
                self.save_checkpoint(step=global_step)
            
            # 达到目标步数则执行最终验证并退出
            if global_step >= total_training_steps:
                self._final_validation()
                return
        
        # 每个 epoch 结束后的验证
        self._validation(epoch)
        self.save_checkpoint(step=global_step)
```

#### 4.2.4 单步训练 (`training_step`)

```python
def training_step(self, batch):
    self.fsdp_model.train()
    self.optimizer.zero_grad()
    
    # 分割为 micro batches
    micro_batches = batch.split(micro_batch_size_per_gpu)
    
    step_loss = 0
    for micro_batch in micro_batches:
        # 前向 + 反向
        loss = self._compute_loss_and_backward(micro_batch) / n_micro_batches
        step_loss += loss.item()
    
    # 梯度裁剪
    grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=clip_grad)
    
    # 检查梯度是否有效
    if torch.isfinite(grad_norm):
        self.optimizer.step()
    
    # 更新学习率
    self.lr_scheduler.step()
    
    # 同步 loss 到所有 DP rank
    torch.distributed.all_reduce(step_loss, op=ReduceOp.AVG)
    
    return {"train/loss": step_loss, "train/lr": lr}
```

#### 4.2.5 Loss 计算与反向传播 (`_compute_loss_and_backward`)

支持两种模式：

**模式 1：标准模式（无序列并行）**
```python
# 标准前向传播
output = self.fsdp_model(input_ids, attention_mask, position_ids)
logits = output.logits

# 计算 cross-entropy loss
shift_logits = logits[:, :-1, :].view(-1, vocab_size)
shift_labels = input_ids[:, 1:].view(-1)
loss = cross_entropy(shift_logits, shift_labels) * loss_mask
```

**模式 2：序列并行模式（Ulysses SP + Remove Padding）**
```python
# 1. Unpad：移除 padding token
input_ids_rmpad, indices, *_ = unpad_input(input_ids, attention_mask)

# 2. 按 SP size 切分输入
input_ids_rmpad_sliced, position_ids_padded, pad_size = \
    ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids, sp_size)

# 3. 前向传播（仅处理有效 token）
output = self.fsdp_model(input_ids_rmpad_sliced, attention_mask=None, position_ids=...)

# 4. 计算 loss
logits_rmpad = output.logits
loss = cross_entropy(logits_rmpad, labels_rmpad)

# 5. Gather 回完整序列
full_loss = pad_input(loss, indices, batch_size, seqlen)
```

### 4.3 数据集类：`SFTDataset`

#### 4.3.1 数据格式支持

| 格式 | 检测方式 | 处理逻辑 |
|------|----------|----------|
| HuggingFace Dataset | 目录格式，无 JSON 文件 | `datasets.load_dataset()` 加载 |
| JSON 文件 | 目录包含 `.json` 文件 | 读取 `input`/`output` 字段 |
| Parquet 文件 | 显式 parquet 文件路径 | `pandas.read_parquet()` 加载 |

#### 4.3.2 单条数据处理流程 (`__getitem__`)

```
┌─────────────────────────────────────────────────────────────┐
│                    __getitem__ 流程                          │
├─────────────────────────────────────────────────────────────┤
│  输入: prompt="写一个 add kernel", response="import triton..."│
│                                                              │
│  1. 构造对话格式                                             │
│     prompt_chat = [{"role": "user", "content": prompt}]     │
│     prompt_str = tokenizer.apply_chat_template(...)         │
│     response_str = response + eos_token                     │
│                                                              │
│  2. Tokenize                                                │
│     prompt_ids = tokenizer(prompt_str)                      │
│     response_ids = tokenizer(response_str)                  │
│                                                              │
│  3. 拼接                                                    │
│     input_ids = concat(prompt_ids, response_ids)            │
│     attention_mask = concat(prompt_mask, response_mask)     │
│                                                              │
│  4. Padding / Truncation (到 max_length)                    │
│     - 短则 pad 到 max_length                                 │
│     - 长则按 truncation 策略截断（left/right/error）          │
│                                                              │
│  5. 构建 loss_mask                                          │
│     - prompt 部分 mask = 0（不参与 loss 计算）               │
│     - response 部分 mask = 1（参与 loss 计算）               │
│     - padding 部分 mask = 0                                  │
│                                                              │
│  输出: {input_ids, attention_mask, position_ids, loss_mask} │
└─────────────────────────────────────────────────────────────┘
```

#### 4.3.3 Loss Mask 构建逻辑

```python
# loss_mask 初始等于 attention_mask
loss_mask = attention_mask.clone()

# 将 prompt 部分（除最后一个 token）的 mask 置 0
if prompt_length > 1:
    loss_mask[:prompt_length - 1] = 0

# 将 response 最后一个 token 的 mask 置 0（EOS token 不计算 loss）
loss_mask[prompt_length + response_length - 1] = 0
```

这样确保只有 **response 的有效 token** 参与 loss 计算。

---

## 五、分布式训练策略

### 5.1 并行策略组合

```
┌─────────────────────────────────────────────────────────────┐
│                    并行策略组合                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  总 GPU 数 = NNODES × GPUS_PER_NODE                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FSDP（数据并行 + 模型分片）                           │   │
│  │  - 模型参数、梯度、优化器状态全部分片                   │   │
│  │  - 每个 GPU 只持有 1/N 的模型状态                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          +                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Ulysses Sequence Parallel（序列并行）                 │   │
│  │  - 将序列维度切分到 SP_SIZE 个 GPU                    │   │
│  │  - 每个 GPU 处理 1/SP_SIZE 的序列长度                 │   │
│  │  - 配合 Flash Attention 的 varlen 接口                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  示例：8 GPU, SP_SIZE=4                                     │
│  - DP Size = 8 / 4 = 2                                      │
│  - 2 个数据并行组，每组 4 个 GPU 做序列并行                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 DeviceMesh 初始化

```python
# FSDP DeviceMesh（全量 GPU 作为一个 FSDP 组）
device_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(world_size,),
    mesh_dim_names=("fsdp",)
)

# Ulysses DeviceMesh（二维 mesh：DP × SP）
dp_size = world_size // sp_size
ulysses_device_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(dp_size, sp_size),
    mesh_dim_names=("dp", "sp")
)
```

### 5.3 数据分发

- **不同 DP 组**：获取不同的数据批次
- **同一 SP 组**：获取相同的数据批次（由 DistributedSampler 保证）
- **SP 组内**：每个 GPU 处理序列的不同片段

---

## 六、配置系统

### 6.1 Hydra 配置

使用 Hydra 进行配置管理，支持命令行覆盖：

```bash
python -m kernel.fsdp_sft_trainer \
  data.train_batch_size=64 \
  data.max_length=18432 \
  model.fsdp_config.cpu_offload=True \
  trainer.total_epochs=4
```

### 6.2 核心配置项

#### 数据配置 (`data`)
```yaml
data:
  train_batch_size: 64          # 全局训练 batch size
  micro_batch_size_per_gpu: 2   # 每张卡的 micro batch
  max_length: 18432             # 最大序列长度
  prompt_key: prompt            # prompt 字段名
  response_key: response        # response 字段名
  truncation: right             # 截断策略
  multiturn:
    enable: true                # 启用多轮对话模式
    messages_key: messages      # 消息列表字段
```

#### 模型配置 (`model`)
```yaml
model:
  partial_pretrain: /path/to/qwen3-8b-base  # 预训练模型路径
  fsdp_config:
    model_dtype: bf16           # 模型精度
    cpu_offload: true           # 启用 CPU offload
    offload_params: false       # 是否 offload 参数（vs 仅 offload 梯度）
  enable_gradient_checkpointing: true  # 梯度检查点
  strategy: fsdp                # 训练策略
```

#### 优化器配置 (`optim`)
```yaml
optim:
  lr: 2e-5                      # 学习率
  betas: [0.9, 0.95]           # AdamW betas
  weight_decay: 0.01            # 权重衰减
  warmup_steps_ratio: 0.1       # warmup 比例
  clip_grad: 1.0                # 梯度裁剪
  lr_scheduler: cosine          # 学习率调度器
```

#### 训练器配置 (`trainer`)
```yaml
trainer:
  project_name: kernel-sft      # WandB 项目名
  experiment_name: drkernel-8b-coldstart  # 实验名
  total_epochs: 4               # 训练轮数
  save_freq: 50                 # 保存频率（step）
  logger: ["console", "wandb"]  # 日志后端
```

---

## 七、Checkpoint 管理

### 7.1 保存逻辑

```python
def save_checkpoint(self, step):
    # 1. 获取完整状态字典（Rank 0 only, offload to CPU）
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = self.fsdp_model.state_dict()
    
    # 2. 保存为 HuggingFace 格式
    path = os.path.join(local_dir, f"global_step_{step}")
    if rank == 0:
        self.model.save_pretrained(path, state_dict=state_dict)
        self.tokenizer.save_pretrained(path)
        
        # 3. 可选：上传到 HDFS
        if hdfs_dir:
            hdfs_io.copy(src=path, dst=hdfs_dir)
```

### 7.2 保存时机
- 每 `save_freq` 个 step 保存一次
- 达到目标训练步数时保存最终 checkpoint
- 每个 epoch 结束时保存

---

## 八、日志与监控

### 8.1 日志层级

| 级别 | 输出位置 | 内容 |
|------|----------|------|
| DEBUG | 文件日志 | 详细调试信息，含文件名/行号 |
| INFO | 控制台 + 文件 | 关键节点信息（配置、进度、指标） |
| WARNING | 控制台 + 文件 | 警告信息 |
| ERROR | 控制台 + 文件 | 错误信息 + traceback |

### 8.2 训练进度日志示例

```
🚀 SFT Training Job Started
📋 完整配置: ...
🔧 关键参数: model=qwen3-8b-base, seq_parallel=4, train_files=...
🖥️  设备类型: cuda
✅ FSDP DeviceMesh 完成 | shape=(8,) | 耗时: 1.23s
✅ Ulysses DeviceMesh 完成 | shape=(2, 4) | 耗时: 0.45s
📥 拷贝模型到本地: /path/to/qwen3-8b-base
✅ 模型拷贝完成 | 耗时: 5.67s
🔤 加载Tokenizer | vocab_size=151936 | 耗时: 2.34s
✅ ChatTemplate已替换为自定义模板
📚 开始构建数据集...
✅ 训练集构建完成 | samples=2000 | 耗时: 12.34s
🎯 初始化FSDP SFT Trainer...
💾 初始化前显存占用: 0.52 GB
✅ Trainer初始化完成 | 耗时: 45.67s | 显存占用: 12.34 GB (峰值: 15.67 GB)
🏃 启动训练流程...
📈 Step 50/320 | Loss: 1.2345 | LR: 1.800e-3 | 耗时: 2.345s
💾 触发检查点保存于 step 50
...
🎉 训练成功完成 | 总耗时: 3600.00s (1.00h)
```

### 8.3 WandB 集成

通过 `Tracking` 类集成 WandB，记录以下指标：
- `train/loss`：训练 loss
- `train/lr(1e-3)`：学习率（缩放后）
- `val/loss`：验证 loss

---

## 九、与 RL 训练的关系

### 9.1 训练流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SFT 阶段    │────▶│  RL 阶段     │────▶│  评估阶段    │
│  (Cold Start)│     │  (PPO/GRPO) │     │  (KernelGYM) │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
  高质量数据           奖励信号              正确性+性能
  监督微调           策略优化              综合评估
```

### 9.2 SFT 与 RL 的共享组件

| 组件 | SFT 使用 | RL 使用 |
|------|----------|---------|
| VERL 框架 | ✅ | ✅ |
| FSDP 分布式 | ✅ | ✅ |
| Ulysses SP | ✅ | ✅ |
| Tokenizer | ✅ | ✅ |
| DeviceMesh | ✅ | ✅ |
| Checkpoint | ✅ | ✅ |

### 9.3 RL 训练器扩展

RL 训练使用 `RayKernelTrainer`（继承自 `RayPPOTrainer`），在 SFT 基础上增加了：
- **多轮 rollout**：支持多次采样和交互
- **奖励计算**：通过 KernelGYM 评估生成 kernel 的正确性和性能
- **优势估计**：支持 GAE、GRPO、RLOO、Optimal Baseline 等多种算法
- **Ray 分布式**：使用 Ray 框架进行分布式 rollout 和训练

---

## 十、关键设计亮点

### 10.1 序列并行 + Remove Padding
- 通过 `unpad_input` 移除 padding token，减少无效计算
- 使用 Ulysses SP 将长序列切分到多个 GPU
- 配合 Flash Attention 的 varlen 接口，提升长序列训练效率

### 10.2 CPU Offload
- 支持将优化器状态和梯度 offload 到 CPU
- 在显存受限时可训练更大的模型
- 通过 `cpu_offload` 和 `offload_params` 配置灵活控制

### 10.3 多格式数据支持
- 自动检测 HuggingFace Dataset、JSON、Parquet 格式
- 支持多轮对话模式（MultiTurnSFTDataset）
- 内置 response token 统计功能，方便数据分析

### 10.4 完善的日志与监控
- 双通道日志（控制台 + 文件）
- RotatingFileHandler 防止磁盘占满
- WandB 集成，支持远程监控
- 训练完成/失败信号文件，方便外部脚本检测

### 10.5 设备兼容性
- 自动检测 NPU / CUDA 设备
- DeviceMesh 初始化适配不同设备类型
- 显存统计仅对 CUDA 设备生效（避免 NPU 报错）

---

## 十一、常见问题与注意事项

### 11.1 Batch Size 配置
- `train_batch_size` 是**全局 batch size**，会被 DP size 整除
- `micro_batch_size_per_gpu` 是每张卡的实际 batch size
- 梯度累积步数 = `train_batch_size / micro_batch_size_per_gpu`

### 11.2 显存优化
- 启用 `gradient_checkpointing` 可显著减少显存（约 50%）
- 启用 `cpu_offload` 可将优化器状态 offload 到 CPU
- 使用 `bf16` 精度相比 `fp32` 减少约 50% 显存

### 11.3 序列长度
- `max_length=18432` 是针对 kernel 生成任务的长序列设置
- 超长序列会触发截断（默认 right truncation）
- 序列并行（SP_SIZE=4）可进一步支持更长序列

### 11.4 Checkpoint 恢复
- 当前版本未实现自动 resume（TODO 注释）
- 手动恢复：加载 HuggingFace 格式的 checkpoint 作为 `partial_pretrain`

---

## 十二、总结

KernelGYM 的 SFT 训练框架基于 VERL 构建，具有以下特点：

1. **轻量设计**：单文件 FSDP SFT Trainer，易于理解和修改
2. **分布式高效**：FSDP + Ulysses SP 组合，支持大规模分布式训练
3. **灵活配置**：Hydra 配置系统，支持命令行动态覆盖
4. **设备兼容**：自动适配 NPU / CUDA 设备
5. **生产就绪**：完善的日志、监控、checkpoint 机制

SFT 阶段产出的模型将作为 RL 训练的起点，通过 KernelGYM 环境的奖励信号进一步优化 kernel 生成能力。
