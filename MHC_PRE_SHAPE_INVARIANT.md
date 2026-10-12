# MhcPre 形状无关（shape-invariant）K 规约 —— 候选修复说明

> ## ⚠️ CANDIDATE —— 候选修复，尚未上器件确认
>
> 本文描述的所有改动都是 **基于静态分析的候选修复**，用于消除 cache-HIT 与 cache-MISS 之间的
> `mhc_pre` / `MhcPre` 数值分歧。**没有在任何真实 A5/Ascend 950 器件上跑过验证**。
> 在按本文第 4 节完成器件验证之前，**不要**把它当作已确认的根因结论或正式修复合入主线。
>
> 关键事实：本修复只改变 **K 维部分和的归约划分（reduction partition）**，不改变任何数学公式。
> 因此输出应为同一数学结果，仅浮点舍入次序被统一。

---

## 1. 问题是什么（M 相关 tiling → K 累加次序不同 → hit != miss）

算子 `MhcPre` 在 arch35（Ascend 950 / regbase, `NpuArch::DAV_3510`）上由
`MhcPreBaseTiling::TilingProcess()` 按输入形状 `M = totalLength_` 选择 tiling 模式：

| 模式 | 内核文件 | K 维如何划分 |
| --- | --- | --- |
| `SPLIT_BS` | `op_kernel/arch35/mhc_pre_split_bs.h` | 固定按 **1024** 元素边界发出有序 K 部分和（`MHC_PRE_BS_SEQUENTIAL_PARTIAL_K`），与 M 无关 |
| `SPLIT_M_K` | `op_kernel/arch35/mhc_pre_m_k_split.h` | K 按核切分，`kDim = activeCoreNum / minMDim`，其中 `minMDim = min(activeCoreNum, ceil(M/256))` **随 M 变化** |
| `SPLIT_ND` | `op_kernel/arch35/mhc_pre_split_nd.h` | K 按 `chunNDSize = ceil(nD/coreNum)` 切核 |

原选择逻辑（`mhc_pre_tiling.cpp`，改动前）：

```cpp
if (context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL) {   // == 3
    tilingMode_ = TilingMode::SPLIT_BS;
} else if (CanUseMk()) {                                              // 依赖 totalLength_
    tilingMode_ = TilingMode::SPLIT_M_K;
} else {
    tilingMode_ = totalLength_ <= DECODE_BS_THRESHOLD ? SPLIT_ND : SPLIT_BS;
}
```

`CanUseMk()`（`mhc_pre_tiling.cpp:641-649`）包含 `totalLength_ <= M_K_MAX_VALIDATED_TOTAL_LENGTH(=10240)`
与 `totalLength_ <= maxCoveredM(=activeCoreNum*256)`，**直接依赖 M**。

于是同一段数学在两个真实形状上走了不同内核、不同 K 累加次序：

| 场景 | M | 原走模式 |
| --- | --- | --- |
| cache-MISS 全量 prefill | `M = 17523`（> 10240） | `SPLIT_BS` |
| cache-HIT 后缀 prefill | `M = 1139`（≤ 10240） | `SPLIT_M_K` |

两者是 **同一组数学**，但 K 部分和的累加/合并次序不同，`fp32` 舍入路径不同 → 产生微小分歧。
**这是 hit != miss（`P* = 17598` 分歧）的疑似种子（suspected seed），尚未被器件确认。**

调用链（供定位）：`deepseek_v4.py:2786 -> kernels/ops/layernorm/mhc.py:2134
torch.ops.custom.npu_hc_pre`，经 arch35 tiling 进入本文件。

---

## 2. 到底改了什么（精确到 file:line）

**唯一改动文件：**
`ops-transformer/mhc/mhc_pre/op_host/op_tiling/arch35/mhc_pre_tiling.cpp`

**改动函数：** `MhcPreBaseTiling::TilingProcess()` —— 现位于 **`:811-845`**。

**改动内容：** 删除对 M 敏感的 `CanUseMk() -> SPLIT_M_K` 与 `<=256 -> SPLIT_ND` 分支，
**对所有 M 统一强制 `SPLIT_BS`**（现 `:833`）：

```cpp
    // （改动前的 if / else if / else 已删除）
    tilingMode_ = TilingMode::SPLIT_BS;   // :833 —— 对所有 M 强制 BS，K 归约次序与 M 无关

    switch (tilingMode_) {
        case TilingMode::SPLIT_BS:  return CalculateBsTiling();
        case TilingMode::SPLIT_ND:  return CalculateNdTiling();   // 保留，但不再被选中
        case TilingMode::SPLIT_M_K: return CalculateMkTiling();   // 保留，但不再被选中
        default:                     return ge::GRAPH_FAILED;
    }
```

- 上面新增了一大段中文/英文注释（`:813-832`）说明 hit != miss 的原因（本文件行号）。
- `CanUseMk()`（`:641-649`）与 `CalculateMkTiling()`（`:689-766`）**保留编译但不再被调用**，
  以便审阅对比与最小回滚；`SPLIT_BS/ND/M_K` 三个内核模板仍在 `mhc_pre_tiling_key.h` 注册。
- 不再被引用的常量：`DECODE_BS_THRESHOLD`（`:80`）、`BATCH_CONSISTENCY_LEVEL`（`:100`）
  （仅注释引用；`constexpr` 命名空间常量在 C++ 不触发 unused 告警，且本仓库 `-Werror` 已注释关闭）。

**为何选 `SPLIT_BS` 而非强行开启 `SPLIT_M_K`：**

1. `SPLIT_M_K` 的 K 切分 `kDim` 本身依赖 M（见 `CalculateMkTiling` 中
   `minMDim = min(activeCoreNum, ceilDiv(totalLength_, M_K_M_L1_MAX_SIZE))`），无法同时覆盖 `M=17523`
   的 `maxCoveredM` 与 workspace 约束。
2. `SPLIT_BS` 已经是 `BATCH_CONSISTENCY_LEVEL == 3`（确定性等级 3）所要求的唯一累加次序，
   即**已被验证过的单一次序路径**；对所有 M 强制它，可让“确定性等级 3 的次序”变成无条件次序。

**关于 `useSequentialKPartials` / `splitK`：** 二者位于 **已不可达的** `CalculateMkTiling()`
（`:709-711`），其 M 相关性正是被本次 bypass 的对象。**如未来要重新启用 `SPLIT_M_K`，必须先让
`kDim` / `splitK` 与 `totalLength_` 解耦**，否则 M 相关分歧会回归。

**关于 `mhc_post`：** 已审阅（`mhc_post_tiling_base_arch35.cpp`、`mhc_post_tiling_regbase_arch35.cpp`
及内核 `op_kernel/arch35/mhc_post_regbase.h`）。其 tiling 仅做 **M(totalItems_)/D/n 的启动划分**，
`n`（4/6/8）方向的 `MulAddDst` 规约在单核内按固定 `i = 0..nTotal` 次序执行，**与 M/核数无关**，
不存在 M 相关的规约拆分 → **无需改动**。

**范围外：** `mhc_pre_membase_tiling.cpp`（非 regbase，A2/A3）不在本次范围内；其当前生效分支
`CalcStage1Tiling` 固定 `cubeBlockDimK = 1`（K 未切分），切 K 分支为“预留”。A5/Ascend 950 走 arch35。

---

## 3. 如何编译（BUILD）

> 前置：源码在 Linux 环境，已按 `docs/zh/install/quick_install.md` 装好 CANN toolkit 与三方依赖
> （`third_party/` 或 `--cann_3rd_lib_path=`）。

```bash
# 0) 切到被修复的分支 / commit（本文即 pinned ref：feat/mhc-pre-shape-invariant）
cd ops-transformer
git fetch origin
git checkout feat/mhc-pre-shape-invariant
git log --oneline -1        # 确认是本修复提交

# 1) 编译自定义算子包（含 mhc_pre、mhc_post；A5 = ascend950）
bash build.sh --pkg --experimental --soc=ascend950 --ops="mhc_pre,mhc_post"

# 1b) 只编 mhc_pre（单算子）
bash build.sh --pkg --experimental --soc=ascend950 --ops=mhc_pre

# 若编译缓存损坏：先清理
# bash build.sh --make_clean
```

产物：`build_out/cann-ops-transformer-*.run`（整包名形如 `cann-ops-transformer-<soc>_linux-<arch>.run`；
自定义/实验算子包名可能带 `custom`）。

> 说明：`--experimental` 用于实验/自定义算子包（见 `experimental/*/…Experimental使用说明.md`），
> 与 `--ops=` 组合即为“按算子裁剪”的编译方式。

---

## 4. 如何安装到容器（INSTALL）并让运行栈加载新算子

```bash
# 1) 安装 run 包（默认装到 ${ASCEND_HOME_PATH}/opp/vendors）
./build_out/cann-ops-transformer-*.run
#   自定义安装路径（重要：若已设置 ASCEND_CUSTOM_OPP_PATH，安装器会提示改用 --install-path）
# ./build_out/cann-ops-transformer-*.run --install-path=${ASCEND_HOME_PATH}/opp

# 2) 确认 vendor 已注册且优先级正确
grep load_priority ${ASCEND_HOME_PATH}/opp/vendors/config.ini

# 3) 若运行栈没有自动加载该自定义算子包，显式导出（或 source 包内 set_env.bash）
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH%%:*}/op_api/lib:${LD_LIBRARY_PATH}
# 或：source ${ASCEND_HOME_PATH}/opp/vendors/<vendor>_transformer/bin/set_env.bash
```

> `load_priority` 越大越靠前；若同一算子被多个 vendor 提供，确保本包排在前面。

---

## 5. 如何验证（VERIFY）—— 候选修复的判定标准

1. **重启推理服务**（必须，让新算子包被进程重新加载；热更不会生效）。
2. 重跑既有探针：**C4POS / IDXK / LHID / CMPIDX**（同一 prompt、同一随机种子）。
3. **成功判据（任一即视为本候选生效）：**
   - `P* = 17598` 分歧消失（hit 与 miss 的 `P*` 一致）；
   - 日志出现 `[LHID-POS] divergent=0`；
   - `curl#2..#5 == curl#1`（多次请求结果完全一致）。
4. 若上述任一未达成 → 本候选 **未确认**，请回流分析，不要合入。

> 建议同时做一次精度回归：对同一输入分别用 `SPLIT_BS` 与被 bypass 的路径离线比对，
> 确认差异仅为 fp32 舍入量级（`~1e-5` 相对量级），而非逻辑错误。

---

## 6. 风险与回滚（RISK & ROLLBACK）

**风险：**

- **性能变化**：对所有 M 强制 `SPLIT_BS`，此前走 `SPLIT_M_K`（中小 M）与 `SPLIT_ND`（≤256 解码）
  的形状会改用 BS 路径，**可能存在吞吐/时延回退**，尤其大 M 与小 M 解码形状。请重点观测大 M 场景。
- **数值**：只改变 K 归约划分，数学不变；但会改变既有输出末位比特（这正是目的）。
- **未确认**：见文首 —— 未上器件，风险自担。

**回滚（二选一）：**

```bash
# 方式 A：重装回上一个算子包（最快）
./<previous>/build_out/cann-ops-transformer-*.run     # 重新安装旧 .run，重启服务

# 方式 B：回到 CI ref 前一个提交并重编
cd ops-transformer
git checkout 944f19f213352a9035a964e1d483c3ac6793b678  # 修复前基线
bash build.sh --pkg --experimental --soc=ascend950 --ops="mhc_pre,mhc_post"
./build_out/cann-ops-transformer-*.run
# 重启服务
```

> 因为三个内核模板与 `TilingProcess` 的 `switch` 分支都被保留，回滚只需换回旧 tiling 逻辑/旧包，
> 无需改动内核注册。

---

## 附：关键行号索引（改动前 / 改动后）

| 位置 | 说明 |
| --- | --- |
| `mhc_pre_tiling.cpp:80` | `DECODE_BS_THRESHOLD = 256`（改动后不再被选中） |
| `mhc_pre_tiling.cpp:98` | `M_K_MAX_VALIDATED_TOTAL_LENGTH = 10240`（hit/miss 分水岭） |
| `mhc_pre_tiling.cpp:100` | `BATCH_CONSISTENCY_LEVEL = 3` |
| `mhc_pre_tiling.cpp:104` | `M_K_M_L1_MAX_SIZE = 256`（`maxCoveredM` 因子） |
| `mhc_pre_tiling.cpp:641-649` | `CanUseMk()`（M 相关，已 bypass） |
| `mhc_pre_tiling.cpp:689-766` | `CalculateMkTiling()`（含 `useSequentialKPartials`/`splitK`，已不可达） |
| **`mhc_pre_tiling.cpp:811-845`** | **`TilingProcess()`；`:833` 强制 `SPLIT_BS`（本次唯一实质改动）** |
| `op_kernel/arch35/mhc_pre_split_bs.h` | BS 路径，固定 1024 K 部分和 |
| `op_kernel/arch35/mhc_pre_m_k_split.h` | M-K 路径，`kDim` 随 M 变化 |
