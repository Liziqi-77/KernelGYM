# SFT流程NPU适配说明文档

本文档记录了将 DrKernel 项目的 SFT (Supervised Fine-Tuning) 训练流程从 GPU 适配到 NPU 的所有修改内容。

---

## 修改概述

本次适配主要涉及以下文件的修改，以支持华为昇腾 NPU 设备：

1. **kernel/fsdp_sft_trainer.py** - SFT训练入口
2. **verl_patch/trainer/code/fsdp_sft_trainer.py** - 核心SFT训练逻辑
3. **verl_patch/workers/code/fsdp_workers.py** - FSDP工作器
4. **verl_patch/utils/random.py** - 随机状态管理
5. **verl_patch/workers/code/actor/dp_actor.py** - Actor数据并行
6. **kernel/scripts/sft/8b-coldstart.sh** - 启动脚本

---

## 详细修改内容

### 1. kernel/fsdp_sft_trainer.py

**修改位置**: 第5行、第15-19行

**修改内容**:
- 添加 `get_device_name` 导入
- 将 `init_device_mesh` 的 `device_type` 参数从硬编码 `"cuda"` 改为动态获取

**代码变更**:
```python
# 添加导入
from verl.utils.device import get_device_name

# 修改前:
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
ulysses_device_mesh = init_device_mesh(
    device_type="cuda", mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp")
)

# 修改后:
device_type = get_device_name()
device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
ulysses_device_mesh = init_device_mesh(
    device_type=device_type, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp")
)
```

---

### 2. verl_patch/trainer/code/fsdp_sft_trainer.py

该文件修改较多，主要包括：

#### 2.1 Flash Attention 导入处理
**位置**: 第34行

**修改内容**:
- 添加条件导入，当设备不支持 flash_attn 时提供 fallback

```python
# 修改前:
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

# 修改后:
try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    # NPU or other devices may not have flash_attn
    index_first_axis = None
    pad_input = None
    rearrange = None
    unpad_input = None
```

#### 2.2 添加设备工具导入
**位置**: 第46-48行

```python
# 添加导入
from verl.utils.device import get_device_name, get_device_id, is_npu_available
```

#### 2.3 Attention 实现类型
**位置**: 第217行

```python
# 修改前:
attn_implementation="flash_attention_2",

# 修改后:
attn_impl = "eager" if is_npu_available() else "flash_attention_2"
attn_implementation=attn_impl,
```

#### 2.4 GPU 内存日志
**位置**: 第199行、第249行、第281行、第290行、第415行、第430行、第439行、第446行

所有 `log_gpu_memory_usage` 调用都添加了 CUDA 设备检查：

```python
# 修改前:
log_gpu_memory_usage("Before model allocation", logger=logger)

# 修改后:
if get_device_name() == "cuda":
    log_gpu_memory_usage("Before model allocation", logger=logger)
```

#### 2.5 FSDP device_id 参数
**位置**: 第276行

```python
# 修改前:
device_id=torch.cuda.current_device(),

# 修改后:
device_id=get_device_id(),
```

#### 2.6 数据移动和 Autocast
**位置**: 第318-326行

```python
# 修改前:
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()
position_ids = batch["position_ids"].cuda()
loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).cuda()
...
with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):

# 修改后:
device = get_device_id()
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
position_ids = batch["position_ids"].to(device)
loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(device)
...
with context, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
```

#### 2.7 Tensor 设备移动
**位置**: 第448行、第511行、第524行、第540行

```python
# 修改前:
step_loss = torch.tensor(step_loss).cuda()
data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()

# 修改后:
step_loss = torch.tensor(step_loss).to(get_device_id())
data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(get_device_id())
```

#### 2.8 Main 函数设备网格
**位置**: 第557-560行

```python
# 修改前:
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

# 修改后:
device_type = get_device_name()
device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
```

---

### 3. verl_patch/workers/code/fsdp_workers.py

#### 3.1 设备网格创建函数
**位置**: 第103-110行

```python
# 修改前:
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        device_mesh = init_device_mesh(
            'cuda', mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=['ddp', 'fsdp']
        )
    return device_mesh

# 修改后:
def create_device_mesh(world_size, fsdp_size):
    device_type = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_type, mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        device_mesh = init_device_mesh(
            device_type, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=['ddp', 'fsdp']
        )
    return device_mesh
```

#### 3.2 Attention 实现类型
**位置**: 第315行、第1128行、第1400行

所有 `flash_attention_2` 都改为条件选择：

```python
# 修改前:
attn_implementation="flash_attention_2"

# 修改后:
attn_implementation="eager" if is_npu_available() else "flash_attention_2"
```

#### 3.3 FSDP device_id
**位置**: 第762行、第1169行、第1263行、第1421行

```python
# 修改前:
device_id=torch.cuda.current_device()

# 修改后:
device_id=get_device_id()
```

#### 3.4 数据设备移动
**位置**: 第1236行、第1259行、第1579行、第1584行

```python
# 修改前:
data = data.to(torch.cuda.current_device())

# 修改后:
data = data.to(get_device_id())
```

#### 3.5 内存统计
**位置**: 第778-779行

```python
# 修改前:
metrics['perf/max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
metrics['perf/max_memory_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)

# 修改后:
if get_device_name() == "cuda":
    metrics['perf/max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
    metrics['perf/max_memory_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)
elif get_device_name() == "npu":
    metrics['perf/max_memory_allocated_gb'] = torch.npu.max_memory_allocated() / (1024**3)
    metrics['perf/max_memory_reserved_gb'] = torch.npu.max_memory_reserved() / (1024**3)
```

#### 3.6 Autocast 设备类型
**位置**: 第1449行

```python
# 修改前:
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):

# 修改后:
with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
```

---

### 4. verl_patch/utils/random.py

**修改位置**: 第1-31行（整个文件重写）

主要修改内容：
- 添加 `get_device_name` 导入
- 根据设备类型（CUDA/NPU）分别调用对应的 RNG 函数
- 统一 RNG 状态字典的键名为 `torch_device`

```python
# 添加导入
from verl.utils.device import get_device_name

# save_random_states 函数
def save_random_states():
    device_name = get_device_name()
    if device_name == 'cuda':
        torch_device_state = torch.cuda.get_rng_state_all()
    elif device_name == 'npu':
        torch_device_state = torch.npu.get_rng_state_all()
    else:
        raise NotImplementedError(f"Unsupported device: {device_name}")
    
    rng_states = {
        'torch_cpu': torch.get_rng_state(),
        'torch_device': torch_device_state,
        'random': random.getstate(),
        'numpy': np.random.get_state(),
    }
    return rng_states

# set_global_seed 函数
def set_global_seed(seed):
    device_name = get_device_name()
    torch.manual_seed(seed)
    if device_name == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device_name == 'npu':
        torch.npu.manual_seed_all(seed)
    else:
        raise NotImplementedError(f"Unsupported device: {device_name}")
    random.seed(seed)
    np.random.seed(seed)

# set_random_states 函数
def set_random_states(rng_states):
    if rng_states is None:
        set_global_seed(42)
    else:
        device_name = get_device_name()
        torch.set_rng_state(rng_states['torch_cpu'])
        if device_name == 'cuda':
            torch.cuda.set_rng_state_all(rng_states['torch_device'])
        elif device_name == 'npu':
            torch.npu.set_rng_state_all(rng_states['torch_device'])
        else:
            raise NotImplementedError(f"Unsupported device: {device_name}")
        random.setstate(rng_states['random'])
        np.random.set_state(rng_states['numpy'])
```

---

### 5. verl_patch/workers/code/actor/dp_actor.py

**修改位置**: 第528行、第530行

```python
# 修改前:
data = {**micro_batch.batch.to(torch.cuda.current_device()), **micro_batch.non_tensor_batch}
data = micro_batch.to(torch.cuda.current_device())

# 修改后:
data = {**micro_batch.batch.to(get_device_id()), **micro_batch.non_tensor_batch}
data = micro_batch.to(get_device_id())
```

---

### 6. kernel/scripts/sft/8b-coldstart.sh

**修改位置**: 第33行后添加

```bash
# Detect device type (NPU or GPU)
if command -v npu-smi &> /dev/null; then
    export DEVICE_TYPE="npu"
    echo "NPU detected, using NPU for training"
else
    export DEVICE_TYPE="cuda"
    echo "GPU detected, using CUDA for training"
fi
```

---

## 依赖要求

### NPU 环境要求

1. **torch_npu**: 必须安装华为昇腾 NPU 的 PyTorch 后端
   ```bash
   pip install torch_npu
   ```

2. **CANN 工具包**: 需要安装华为 CANN (Compute Architecture for Neural Networks) 工具包

3. **NPU 驱动**: 确保 NPU 驱动已正确安装

### 启动命令

在 NPU 环境中启动训练：

```bash
# 进入项目目录
cd /home/liziqi/agent/KernelGYM/drkernel

# 运行启动脚本（会自动检测 NPU）
bash kernel/scripts/sft/8b-coldstart.sh
```

或者手动指定设备类型：

```bash
export DEVICE_TYPE=npu
bash kernel/scripts/sft/8b-coldstart.sh
```

---

## 注意事项

1. **Flash Attention**: NPU 设备不支持 `flash_attention_2`，代码会自动回退到 `eager` 模式

2. **内存监控**: GPU 内存监控工具（如 `log_gpu_memory_usage`）仅在 CUDA 设备上可用，NPU 环境会自动跳过

3. **性能差异**: 由于 NPU 不支持 Flash Attention，训练速度可能会比 GPU 慢

4. **分布式训练**: 确保 NPU 环境中的分布式训练配置正确（HCCL 后端）

5. **RNG 状态**: 随机数生成器状态现在统一存储在 `torch_device` 键下，不再区分 `torch_cuda`

---

## 测试验证

在 NPU 环境中验证适配是否成功：

```python
import torch
import torch_npu

# 检查 NPU 是否可用
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")

# 检查 verl 设备工具
from verl.utils.device import get_device_name, get_device_id
print(f"Device name: {get_device_name()}")
print(f"Device ID: {get_device_id()}")
```

---

## 附录：修改文件清单

| 序号 | 文件路径 | 修改类型 | 优先级 |
|------|---------|---------|--------|
| 1 | kernel/fsdp_sft_trainer.py | device mesh 初始化 | 高 |
| 2 | verl_patch/trainer/code/fsdp_sft_trainer.py | 核心 SFT 训练逻辑 | 高 |
| 3 | verl_patch/workers/code/fsdp_workers.py | FSDP workers | 高 |
| 4 | verl_patch/utils/random.py | RNG 状态管理 | 中 |
| 5 | verl_patch/workers/code/actor/dp_actor.py | Actor 设备 | 中 |
| 6 | kernel/scripts/sft/8b-coldstart.sh | 启动脚本 | 低 |

---

## 联系方式

如有问题或需要进一步支持，请联系项目维护团队。
