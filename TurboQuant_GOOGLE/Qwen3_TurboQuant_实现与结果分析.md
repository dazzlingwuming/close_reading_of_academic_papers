# Qwen3-1.7B 上的 TurboQuant KV Cache 压缩实现与结果分析

## 1. 背景

在自回归大模型推理中，`KV cache` 是长上下文场景下最主要的显存瓶颈之一。

以 `Qwen3-1.7B` 为例，随着上下文长度增长，每一层每一个历史 token 都要保存对应的 `key/value` 表示。对于长上下文推理：

- 模型权重占用基本固定
- `KV cache` 会随着上下文长度线性增长
- 当上下文从 `2K` 增长到 `8K / 16K / 32K` 时，显存和推理延迟都会快速恶化

TurboQuant 的目标就是在不明显破坏质量的前提下，对 `KV cache` 做在线压缩，并尽量直接在压缩表示上完成 attention 计算。

本文档记录了在 `Qwen3-1.7B` 上实现 TurboQuant 的过程、当前代码结构，以及 benchmark 结果分析。

---

## 2. TurboQuant 的核心思想

TurboQuant 论文不是简单做“逐元素量化后再反量化”，而是针对内积估计设计了两阶段量化器。

### 2.1 两阶段量化

论文中的 `TurboQuant_prod` 对每个向量 `x` 进行如下压缩：

1. 先做 `b-1` bit 的 `TurboQuant_mse`
2. 得到 MSE 重构结果 `x_mse`
3. 计算残差 `r = x - x_mse`
4. 对残差做 `1-bit QJL`
5. 最终存储：
   - `idx`: MSE 量化索引
   - `qjl`: 残差的符号压缩表示
   - `gamma`: 残差范数

论文算法 2 的形式可以写成：

```text
Qprod(x) = [Qmse(x), Qqjl(x - Qmse^{-1}(Qmse(x))), ||x - Qmse^{-1}(Qmse(x))||_2]
```

### 2.2 为什么它适合 KV cache

attention 的关键不是逐元素恢复 `K/V`，而是：

- `q · k` 决定注意力分数
- `softmax(qk) · v` 决定最终输出

因此只要压缩表示能够稳定估计这些量，就不需要保存完整高精度 `K/V`。

这也是 TurboQuant 和普通“量化后再还原”的根本区别：

- 普通量化：更关注向量逐元素恢复误差
- TurboQuant：更关注内积和几何结构保持

---

## 3. 在 Qwen3-1.7B 上的实现目标

这次实现的目标不是“简单存低比特索引”，而是：

1. 用 TurboQuant 压缩 `Qwen3` 的 `KV cache`
2. 在 attention 阶段尽量直接使用压缩表示
3. 避免“整段历史 K/V 全量反量化回 fp16”
4. 对长上下文做 benchmark，观察显存变化

实现代码主要位于：

- [modeling.py](/D:/github/close_reading_of_academic_papers/TurboQuant_GOOGLE/my_qwen3/modeling.py)
- [configuration.py](/D:/github/close_reading_of_academic_papers/TurboQuant_GOOGLE/my_qwen3/configuration.py)
- [benchmark_turboquant.py](/D:/github/close_reading_of_academic_papers/TurboQuant_GOOGLE/benchmark_turboquant.py)

---

## 4. 代码结构概览

### 4.1 配置扩展

在 [configuration.py](/D:/github/close_reading_of_academic_papers/TurboQuant_GOOGLE/my_qwen3/configuration.py) 中为 `Qwen3Config` 增加了：

- `use_turboquant`
- `turboquant_bits`
- `turboquant_block_size`

这些参数分别控制：

- 是否启用 TurboQuant
- 总 bit 数
- attention 按 block 处理时的块大小

### 4.2 TurboQuantCache

核心缓存结构在 [modeling.py](/D:/github/close_reading_of_academic_papers/TurboQuant_GOOGLE/my_qwen3/modeling.py) 的 `TurboQuantCache`。

这个类负责：

1. 保存每层的旋转矩阵 `rot_mat`
2. 保存每层的 `codebook`
3. 保存每层的 QJL 投影矩阵 `proj_mat`
4. 分别压缩并保存 `K/V` 的：
   - `idx`
   - `qjl sign`
   - `gamma`
   - `norm`

缓存里分别维护：

- `key_mse_cache`
- `key_qjl_cache`
- `key_gamma_cache`
- `key_norm_cache`
- `value_mse_cache`
- `value_qjl_cache`
- `value_gamma_cache`
- `value_norm_cache`

### 4.3 压缩步骤

对于每个新进入 cache 的 `key/value`：

1. 先做归一化，记录原始 `norm`
2. 用随机正交矩阵 `Π` 旋转
3. 对旋转后坐标做 `b-1` bit 标量量化，得到 `idx`
4. 由 `idx` 得到 `mse` 重构
5. 计算残差 `r`
6. 用 `S` 做 QJL 投影并取符号，得到 `qjl sign`
7. 存储残差范数 `gamma`

也就是说，当前每个缓存向量最终不再保存成原始 `fp16 K/V`，而是保存为 TurboQuant 的压缩表示。

---

## 5. Attention 是怎么接入的

### 5.1 Qwen3 原始 attention

Qwen3 原始 attention 路径本质上是：

```text
attn_weights = Q @ K^T
attn_output  = softmax(attn_weights) @ V
```

标准实现默认要求拿到完整浮点 `K/V`。

### 5.2 本实现的 TurboQuant attention

当前 TurboQuant attention 位于 [modeling.py](/D:/github/close_reading_of_academic_papers/TurboQuant_GOOGLE/my_qwen3/modeling.py) 的 `turboquant_attention_forward`。

它的主要流程是：

1. 将查询 `Q` 旋转到 TurboQuant 的坐标系
2. 读取某个 block 的压缩 cache 表示
3. 用压缩 `idx/sign/gamma/norm` 直接计算该 block 的 logits
4. 通过 online softmax 进行数值稳定累积
5. 对 `V` 侧同样根据压缩表示做聚合
6. 将 block 结果累计成最终输出

### 5.3 为什么说它“更接近压缩域计算”

前面的错误实现是：

- 先把整个 block 的 `K/V` 恢复成普通浮点向量
- 再做普通 attention

现在这版已经去掉了这条路径：

- `get_prod_block()` 返回的是压缩表示，而不是完整向量
- attention 里基于 `idx/sign/gamma/norm` 直接求分数和加权和
- 不再构造“整段历史完整 K/V 浮点缓存”

需要说明的是：

严格意义上的“纯压缩域 kernel”通常还会进一步避免中间查表结果的显式 materialize，并做 fused kernel 优化。当前版本已经从算法路径上接近压缩域 attention，但工程上仍是 PyTorch 原型，而不是自定义 CUDA kernel。

---

## 6. 为了提速做过哪些优化

在实现过程中，最初版本虽然逻辑上接近论文，但速度极差。主要原因是：

- Python 层 per-level 循环
- 逐维 unpack `idx`
- 逐维 unpack `sign`
- 小块碎片化计算过多

后续做了以下改动：

### 6.1 去掉 per-level Python 循环

之前 `MSE` 部分是按每个 codebook level 构造 mask 后逐级累加，GPU 吞吐很差。

现在改成：

- 直接 `codebook[idx]`
- 一次性得到 centroid 张量
- 再统一做 `matmul`

### 6.2 `idx` 和 `sign` 解包改为张量化

不再按 128 个维度逐列循环，而是用向量化 bit 操作一次展开。

### 6.3 block-wise online softmax

attention 不是一次构造完整超大矩阵，而是按 block 流式处理，减小长上下文时的峰值显存。

这些改动之后，速度从“几乎不可运行”下降到了“仍然慢，但已能完成 benchmark”。

---

## 7. Benchmark 设置

benchmark 脚本是 [benchmark_turboquant.py](/D:/github/close_reading_of_academic_papers/TurboQuant_GOOGLE/benchmark_turboquant.py)。

测试方式：

- 模型：`Qwen3-1.7B`
- Prompt 长度：`2048 / 4096 / 8192`
- 生成 token 数：`8`
- 对比：
  - `Baseline fp16 KV`
  - `TurboQuant 3-bit KV`

输出指标包括：

- `PromptTok`
- `GenTok`
- `Prefill(s)`
- `Gen(s)`
- `PrefillMem`
- `PeakMem`
- `EndMem`

其中：

- `Prefill(s)`：输入长 prompt 时的前向耗时
- `Gen(s)`：基于 KV cache 继续生成 8 个 token 的耗时
- `PeakMem`：该轮测试中的 GPU 峰值显存

这里的 `Gen(s)` 是生成阶段耗时，不是“向量解码时间”。

---

## 8. 实验结果

### 8.1 2K 长度

```text
Baseline fp16 KV       2082  8  0.35s  0.22s  4122.63MB  4245.37MB  4123.79MB
TurboQuant 3-bit KV    2082  8  0.95s  5.24s  3955.12MB  3980.33MB  3955.57MB
```

观察：

- 峰值显存下降约 `265 MB`
- 速度明显变慢

结论：

- 在 2K 上下文时，权重仍占显存大头
- KV cache 压缩收益已经出现，但还不够显著
- 速度损失相对更容易暴露

### 8.2 4K 长度

```text
Baseline fp16 KV       4129  8  0.69s  0.83s  4938.99MB  6376.65MB  4940.15MB
TurboQuant 3-bit KV    4129  8  2.22s  8.00s  4595.55MB  4620.76MB  4595.97MB
```

观察：

- 峰值显存下降约 `1756 MB`
- 显存收益开始明显放大
- 速度仍慢于 baseline

结论：

- 当上下文达到 4K 左右，KV cache 已成为较大显存来源
- TurboQuant 的长上下文优势开始体现出来

### 8.3 8K 长度

```text
Baseline fp16 KV       8223  8  45.44s  61.03s  6573.25MB  14190.60MB  6575.22MB
TurboQuant 3-bit KV    8223  8  7.13s   11.64s  5873.59MB  5905.71MB   5878.44MB
```

观察：

- 峰值显存下降约 `8285 MB`
- TurboQuant 明显快于当前 baseline

这是最关键的一组结果。

它说明：

- 在超长上下文下，原始路径的 attention 显存和时间都会急剧爆炸
- TurboQuant 的 block-wise 压缩 attention 能有效控制住这部分开销

但这里也要谨慎解释：

- 当前 baseline 并不一定是最优实现
- 长上下文下 baseline 很可能走了一条非常重的普通 attention 路径
- 因此这个“速度大幅领先”更准确地说是“相对于当前实现的 baseline”

它可以证明 TurboQuant 在当前工程中对长上下文有强收益，但不能直接等价为“对任何优化良好的 FP16 baseline 都有相同倍率提升”

---

## 9. 结果怎么理解

### 9.1 是否真的省显存

是的，结论非常明确。

而且上下文越长，节省越明显：

- 2K：下降约 `265 MB`
- 4K：下降约 `1.76 GB`
- 8K：下降约 `8.29 GB`

这非常符合 KV cache 压缩的理论预期。

### 9.2 是否真的提升速度

短上下文下不一定。

在 `2K / 4K` 里：

- TurboQuant 路径仍然更复杂
- 压缩域计算的开销可能超过节省下来的带宽收益

在 `8K` 里：

- 原始 attention 路径的复杂度和显存代价急剧上升
- TurboQuant 的 block-wise 方案开始体现优势

所以速度结论不能简单概括成“总是更快”，更准确的表述是：

- 短上下文：不一定占优
- 长上下文：优势显著增强

### 9.3 目前实现是否已经达到论文里的最佳工程状态

还没有。

目前版本更接近“论文思路的 PyTorch 原型实现”，而不是最终高性能部署版。

距离真正部署级实现，还差：

1. 更彻底的 fused kernel
2. 更高效的 bit-packed 读取
3. 更少的中间张量 materialization
4. 更公平的 baseline 对照

---

## 10. 当前实现的优点与不足

### 优点

- 已经把 TurboQuant 的 `MSE + QJL` 两阶段思想接入 `Qwen3`
- 已经不再走“整段历史 K/V 全量反量化”的错误路径
- 能够在长上下文下明显降低显存占用
- 在当前工程中，8K 上下文下表现出非常强的长上下文优势

### 不足

- 当前仍是 PyTorch 原型，尚非 CUDA kernel 级优化
- 短上下文下速度不占优
- benchmark 中 baseline 可能不是最优可比实现
- 论文中关于质量指标的部分，这里尚未完整复现实验，如：
  - Needle Recall@1
  - LongBench score
  - 质量-压缩比系统评估

---

## 10.5 常见误区：是否应该维护整段浮点历史 KV

一个很自然的想法是：

> 既然当前实现解码阶段较慢，是否可以把历史 `KV` 先解量化成浮点形式缓存起来，后续每步只量化新 token，然后直接复用这些浮点历史？

这个想法只对了一半。

### 正确的部分

如果某个实现真的在每一步都：

1. 读取整段历史压缩 `KV`
2. 把整段历史完整还原为浮点 `K/V`
3. 再做普通 attention

那么它确实会导致：

- 解码时间开销很大
- 每步重复做大量无效工作
- 序列越长，速度越差

这类实现方式显然是不理想的。

### 错误的部分

但是，“把整段历史 `KV` 直接缓存成浮点”并不是正确修复方式。

原因是：

1. 这会把已经压下去的 `KV cache` 显存重新吃回来
2. TurboQuant 的目标本来就是减少历史 `KV` 的常驻显存
3. 如果浮点历史完整常驻，那么 TurboQuant 压缩的意义会被大幅削弱

换句话说：

- 保存整段浮点历史 `KV`，本质上是在“用显存换速度”
- 这和 TurboQuant 的长上下文部署目标是冲突的

### 更准确的理解

正确目标不是：

- “保存完整浮点历史 `KV`”

而是：

- “避免重复重建历史 `KV`”
- “直接在压缩表示上完成 attention 计算”
- “尽量减少中间张量的 materialization”
- “把压缩域 attention 做成高效张量化甚至 fused kernel”

因此，更准确的结论应该是：

> 当前实现的主要问题，不是“没有维护浮点历史缓存”，而是“压缩域 attention 还不够高效”。  
> 如果通过保存完整浮点历史来换速度，会直接损失 TurboQuant 最核心的显存优势。

### 工程上更合理的优化方向

真正合理的方向包括：

1. 保持压缩表示常驻
2. 在 attention 前向中尽量直接使用 `idx / sign / gamma / norm`
3. 避免把整段 `K/V` 重建为普通浮点缓存
4. 使用更高效的张量化实现或 fused kernel 减少运行时开销

这才是既保留 TurboQuant 显存优势、又继续优化速度的正确路线。

---

## 11. 后续优化方向

如果要进一步逼近论文里“部署级效果”，建议继续做：

### 11.1 实现 fused 压缩域 attention kernel

当前最大的提升空间就在这里。

目标是：

- 压缩表示读取
- MSE 项查表
- QJL 项估计
- online softmax
- value aggregation

尽量在更少 kernel 中完成。

### 11.2 做更公平的 baseline

需要和更强的 FP16 / FlashAttention / blockwise baseline 对比，否则速度结论容易受实现细节影响。

### 11.3 复现论文质量指标

仅仅看显存和速度还不够，还应补：

- Needle-In-A-Haystack
- LongBench
- 不同 bit-width 下质量变化

### 11.4 支持 outlier-aware mixed precision

论文和相关工作中，常通过：

- outlier channel 更高 bit
- regular channel 更低 bit

实现更优的质量-压缩平衡。

---

## 12. 总结

本次在 `Qwen3-1.7B` 上实现的 TurboQuant，已经从最初错误的“量化后整段反量化”版本，推进到了更接近论文 `TurboQuant_prod` 的压缩域 attention 原型。

从结果看：

- 短上下文下，显存有下降，但速度未必占优
- 长上下文下，显存收益非常明显
- 在当前工程实现中，`8K` 长度已经显示出非常强的优势

因此可以得出一个较稳妥的结论：

> 对 `Qwen3-1.7B` 而言，TurboQuant 非常适合长上下文 KV cache 压缩；当上下文增长到一定规模后，它能显著降低显存占用，并在当前实现下表现出明显的长上下文推理优势。

但如果目标是进一步达到论文图表里那种“更高质量、更高吞吐、更稳定”的部署效果，仍然需要继续向 fused kernel 和系统级优化推进。
