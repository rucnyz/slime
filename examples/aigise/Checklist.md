# AIgiSE 集成到 RL 框架的开发清单

## 任务需求

将 AIgiSE 项目的 agent 功能集成到多种 RL 框架的 rollout 系统中（slime, verl, areal 等），使得：
1. RL 框架可以调用 `aigise` 提供的统一接口
2. 通过 URL 参数配置 LiteLLM 后端
3. 支持多轮对话的 agent 生成

---

## 架构设计

### 设计原则

AIgiSE 作为通用插件，为多个 RL 框架提供 agent 能力。采用 **适配器模式** 实现框架解耦：

```
RL Framework (slime/verl/areal)
         │
         ▼
    aigise.create()
         │
         ▼
      Client ─────► Evaluation 实例 (benchmark 特定逻辑，初始化一次)
         │                │
         ▼                ▼
     RLSession      _create_task() + _generate_sample()
         │
         ▼
   Framework Adapter ─► 格式转换
   ┌─────┴─────┐        slime Sample ↔ dict ↔ EvaluationTask
   │           │
SlimeAdapter  VerlAdapter  ArealAdapter (TODO)
```


**核心流程**：
1. `Client` 初始化时创建 `Evaluation` 实例（只做一次）
2. 每个 sample 的处理流程：
   - Adapter 将 RL 框架的 Sample 转为 dict 格式
   - 调用 `Evaluation._create_task(dict)` 创建 EvaluationTask
   - 调用 `Evaluation._generate_sample(task)` 运行 agent
   - Adapter 将结果转回 RL 框架的 Sample 格式

### AIgiSE 接口设计

```python
import aigise

# 1. 创建客户端
client = aigise.create(agent_name, benchmark_name, api_base, model_name)

# 2. 初始化会话
with client.init_session() as session:
    # 3. 调用框架特定的 generate 方法
    sample = await session.slime_generate(args, sample, sampling_params)
    # 或
    sample = await session.verl_generate(args, sample, sampling_params)
    # 或
    sample = await session.areal_generate(args, sample, sampling_params)
```

---

## AIgiSE 端实现

### 文件结构

```
AIgiSE/
├── src/aigise/
│   ├── __init__.py              # 导出 aigise.create(), Client, RLSession
│   ├── rl_integration/          # RL 框架集成模块
│   │   ├── __init__.py
│   │   ├── client.py            # Client, RLSession, create()
│   │   ├── benchmark_interface.py  # BenchmarkInterface 类
│   │   └── adapters/            # 框架适配器
│   │       ├── __init__.py
│   │       ├── base.py          # BaseAdapter (抽象基类)
│   │       └── slime.py         # SlimeAdapter
│   ├── evaluations/             # benchmark 定义 (RL 集成接口)
│   │   ├── __init__.py          # Evaluation 基类, 注册表, RL 集成方法
│   │   ├── secodeplt/
│   │   │   └── vul_detection.py # SeCodePLT 评估类 (自动注册)
│   │   └── cybergym/
│   │       └── cybergym_static.py # CyberGym 评估类 (自动注册)
│   └── session/
│       └── aigise_session.py    # 核心 AigiseSession 类
└── examples/agents/             # agent 定义
    └── vul_agent_static_tools/
        └── agent.py
```

### 核心组件

#### 1. `aigise.create()` - 入口函数

```python
def create(
    agent_name: str,      # examples/agents/ 或 aigise/agents/ 下的 agent 名称
    benchmark_name: str,  # evaluations/ 下的 benchmark 名称
    api_base: str,        # LLM API 地址 (从 RL 框架传入)
    model_name: str,      # 模型名称 (从 RL 框架传入)
) -> Client:
```

#### 2. `Client` - 客户端类

- 解析 agent 目录路径
- 加载 `BenchmarkInterface` 和创建 `Evaluation` 实例
- 提供 `init_session()` 创建会话

#### 3. `RLSession` - 会话类

- 包装 `AigiseSession`，复用其资源管理能力
- 提供框架特定的 generate 方法：
  - `slime_generate()` - 已实现
  - `verl_generate()` - TODO
  - `areal_generate()` - TODO
- 支持上下文管理器，自动清理资源

#### 4. 适配器 (Adapters)

| 适配器 | 文件 | 状态 | 说明 |
|--------|------|------|------|
| `BaseAdapter` | `adapters/base.py` | 已完成 | 抽象基类，定义适配器接口 |
| `SlimeAdapter` | `adapters/slime.py` | 已完成 | slime 框架适配器 |
| `VerlAdapter` | - | TODO | verl 框架适配器 (待创建) |
| `ArealAdapter` | - | TODO | areal 框架适配器 (待创建) |

**适配器职责**：
- `convert_to_sample_dict(sample)`: 将 RL 框架的 Sample 转为 dict 格式
- `generate(args, sample, sampling_params)`: 调用 Evaluation 处理 sample
- `update_sample_success/error()`: 将结果转回 RL 框架格式

**适配器不再负责**：
- ~~运行 agent~~ (由 `Evaluation._generate_sample` 处理)
- ~~创建 Runner/SessionService~~ (由 Evaluation 处理)

#### 5. `BenchmarkInterface` - Benchmark 接口

`BenchmarkInterface` 负责加载和封装 benchmark 特定的逻辑，使适配器可以使用统一的接口访问不同 benchmark 的功能。

```python
from aigise.rl_integration import BenchmarkInterface

# 从注册表加载（Evaluation 子类自动注册）
interface = BenchmarkInterface.load("secodeplt")

# 获取 Evaluation 类
eval_class = interface.evaluation_class  # SeCodePLT

# 调用类方法（直接绑定到 Evaluation 子类的方法）
prompt = interface.get_prompt(sample)
reward = await interface.reward_func(args, sample)
```

**加载机制：**

`BenchmarkInterface.load()` 会：
1. 导入 benchmark 子模块以触发 `Evaluation` 子类的注册
2. 从注册表中查找对应的 `Evaluation` 子类
3. 直接绑定该类的 RL 集成方法

**自动注册机制：**

所有 `Evaluation` 子类通过 `__init_subclass__` 自动注册到全局注册表：

```python
# aigise/evaluations/__init__.py
@dataclass
class Evaluation(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        _EVALUATION_REGISTRY[name] = cls

# aigise/evaluations/secodeplt/vul_detection.py
@dataclass
class SeCodePLT(Evaluation):  # 自动注册为 "secodeplt"
    ...
```

注册表 API：
```python
from aigise.evaluations import get_evaluation_class, list_evaluations

# 按名称查找（不区分大小写）
cls = get_evaluation_class("secodeplt")  # -> SeCodePLT

# 列出所有注册的 benchmark
names = list_evaluations()  # -> ["secodeplt", "cybergym", ...]
```

**Evaluation 类的 RL 集成方法：**

| 方法 | 必需 | 说明 |
|------|------|------|
| `get_prompt(cls, sample) -> str` | 否 | 从 RL sample 中提取 prompt (classmethod) |
| `reward_func(cls, args, sample, **kwargs) -> dict` | 否 | 计算 reward (async classmethod) |
| `preprocess_sample(cls, sample) -> sample` | 否 | 预处理 sample (classmethod) |
| `postprocess_response(cls, sample, response) -> sample` | 否 | 后处理 response (classmethod) |

这些方法定义在 `Evaluation` 基类中（有默认实现），子类可以 override：

```python
@dataclass
class SeCodePLT(Evaluation):
    ...

    @classmethod
    def get_prompt(cls, sample: Any) -> str:
        """SeCodePLT-specific prompt extraction."""
        if hasattr(sample, "metadata") and "basedir" in sample.metadata:
            return f"The code is in {sample.metadata['basedir']}."
        return super().get_prompt(sample)

    @classmethod
    async def reward_func(cls, args: Any, sample: Any, **kwargs) -> dict:
        """SeCodePLT-specific reward calculation."""
        ...
```

---

## slime 端实现

### 需要在 slime 项目中创建的文件

**文件**: `examples/aigise/generate_with_aigise.py`

```python
import os
import aigise
from slime.utils.types import Sample

_aigise_client = None

def _get_client(api_base: str, model_name: str):
    global _aigise_client
    if _aigise_client is None:
        benchmark_name = os.getenv("AIGISE_BENCHMARK_NAME", "secodeplt")
        agent_name = os.getenv("AIGISE_AGENT_NAME", "vul_agent_static_tools")
        _aigise_client = aigise.create(agent_name, benchmark_name, api_base, model_name)
    return _aigise_client

async def generate(args, sample: Sample, sampling_params) -> Sample:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    client = _get_client(url, args.model_name)
    with client.init_session() as session:
        sample = await session.slime_generate(args, sample, sampling_params)
    return sample
```

### slime 端需要做的事情

1. **安装 aigise 依赖**

   **方式一：从 Git 仓库安装（推荐用于开发）**
   ```bash
   # 使用 pip
   pip install git+https://github.com/anthropics/AIgiSE.git

   # 或使用 uv
   uv pip install git+https://github.com/anthropics/AIgiSE.git
   ```

   **方式二：本地开发模式安装**
   ```bash
   # 使用 pip 安装（可编辑模式）
   pip install -e .

   # 或使用 uv 安装
   uv pip install -e .
   ```

   **注意事项：**
   - AIgiSE 需要 Python >= 3.12
   - 主要依赖：`google-adk>=1.18.0`, `litellm>=1.74.7`

2. **运行 rollout**
   ```bash
   # 设置环境变量
   export AIGISE_AGENT_NAME="vul_agent_static_tools"
   export AIGISE_BENCHMARK_NAME="secodeplt"

   # 运行 slime rollout
   python -m slime.rollout ... \
       --generate_func examples.aigise.generate_with_aigise.generate
   ```

---

## 环境变量配置

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `AIGISE_AGENT_NAME` | AIgiSE 中 `examples/agents/` 目录下的 agent 名称 | `vul_agent_static_tools` |
| `AIGISE_BENCHMARK_NAME` | AIgiSE 中 `evaluations/` 目录下的 benchmark 名称 | `secodeplt` |

---

## 架构图

```
slime/verl/areal rollout
    │
    ├── generate(args, sample, sampling_params)
    │       │
    │       ├── aigise.create(agent_name, benchmark_name, api_base, model_name)
    │       │       ├── 解析 agent_dir
    │       │       ├── 加载 BenchmarkInterface
    │       │       └── 创建 Evaluation 实例 (初始化一次)
    │       │
    │       ├── client.init_session()
    │       │       └── 返回 RLSession (上下文管理器)
    │       │
    │       └── session.slime_generate(args, sample, sampling_params)
    │               │
    │               ├── 获取/创建 SlimeAdapter
    │               ├── adapter.convert_to_sample_dict(sample) → dict
    │               ├── evaluation._create_task(dict) → EvaluationTask
    │               ├── evaluation._generate_sample(task) → result
    │               └── adapter.update_sample_success(sample, result)
    │
    └── 继续训练流程
```

---

## 开发进度

### 已完成

- [x] 设计适配器模式架构
- [x] 创建 `rl_integration/` 目录结构
  - [x] `rl_integration/__init__.py` - 模块导出
  - [x] `rl_integration/client.py` - Client, RLSession, create()
  - [x] `rl_integration/benchmark_interface.py` - BenchmarkInterface
  - [x] `rl_integration/adapters/base.py` - BaseAdapter
  - [x] `rl_integration/adapters/slime.py` - SlimeAdapter
- [x] 实现 `BaseAdapter` 抽象基类
- [x] 实现 `SlimeAdapter`
- [x] 重构 `Client` 和 `RLSession`
- [x] 更新 `aigise/__init__.py` 导出
- [x] 实现 Evaluation 自动注册机制
  - [x] 在 `Evaluation` 基类中添加 `__init_subclass__` 自动注册
  - [x] 创建 `_EVALUATION_REGISTRY` 注册表
  - [x] 实现 `get_evaluation_class()` 和 `list_evaluations()` API
- [x] 实现 RL 集成方法作为 Evaluation 类方法
  - [x] `Evaluation.get_prompt(cls, sample)` - 基类默认实现
  - [x] `Evaluation.reward_func(cls, args, sample)` - 基类默认实现
  - [x] `SeCodePLT.get_prompt()` - secodeplt 特定实现
  - [x] `SeCodePLT.reward_func()` - secodeplt 特定实现
- [x] 实现 `BenchmarkInterface` 机制
  - [x] 从注册表加载 Evaluation 子类
  - [x] 直接绑定类方法到 interface
- [x] 将 `evaluations/` 移到 `src/aigise/evaluations/`
- [x] 重构架构：Adapter 使用 Evaluation._generate_sample
  - [x] `Client._load_benchmark` 创建 Evaluation 实例
  - [x] `BaseAdapter` 接收 Evaluation 实例
  - [x] `SlimeAdapter` 实现 `convert_to_sample_dict()`
  - [x] `SlimeAdapter.generate()` 调用 `evaluation._create_task()` 和 `_generate_sample()`
  - [x] 删除 `SlimeAdapter._run_agent()` (由 Evaluation 处理)

### 待完成

- [ ] 实现 `VerlAdapter`
- [ ] 实现 `ArealAdapter`
- [ ] 端到端测试

---

## 注意事项

1. **依赖**: slime 需要安装 `aigise` 包
2. **异步**: 所有 `*_generate` 方法都是异步方法
3. **职责分离**:
   - RL 框架只负责调用接口，传递参数
   - AIgiSE 负责所有 agent 逻辑、工具调用、多轮对话
4. **扩展性**: 添加新框架只需：
   - 创建新的 Adapter 类继承 `BaseAdapter`
   - 在 `RLSession` 中添加对应的 `*_generate` 方法
