# quant_lightning_indexer_metadata：supportFd_ 形状无关修复

> 基线：release tag **`v9.1.0`** = `c77c0829f1f0f0b8f882db1d9b051a758b605472`（对应设备 CANN 9.1.0）
> 修复分支：`feat/indexer-supportfd-shape-invariant`
> 状态：**CANDIDATE（代码层结论，尚未在设备上验证）**

---

## (a) 问题

设备运行的是从本 tag 构建的 `custom_transformer` vendor 算子（覆盖 CANN 内置实现）。
现象属于 “indexer top-k 随 shape 变化” 一类：同一段逻辑序列在不同 shape（不同 M / query 长度）
下，被选中的 top-k 稀疏索引不一致，进而喂给 sparse attention，导致下游不稳定。

根因怀疑点在 **indexer metadata（AICPU 侧）**：

`experimental/attention/quant_lightning_indexer_metadata/op_kernel_aicpu/quant_lightning_indexer_metadata_aicpu.cpp`
的 `CalcSplitInfo()`（v9.1.0 行 159–189）里，`supportFd_` 由 `maxS2Size_` 决定：

- `maxS2Size_ = max over batches( S2 / cmpRatio_ )`，`cmpRatio_` 默认 4（`CalcSplitInfo` 行 165）
- 翻转条件：
  - **行 175**：`if (maxS2Size_ > sparseCount_)` → `supportFd_ = true;`（`sparseCount_` 默认 **512**，来自属性 `sparse_count`）
  - **行 180–183**（Ascend910B）：`maxS2Size_ > 2 * s2BaseSize_` (= 2×2048 = 4096) → true
  - **行 184–187**（Ascend950 / A5）：`maxS2Size_ > 5 * s2BaseSize_` (= 5×128 = **640**) → true
- 由于行 175 先命中并 `return`，在 A5 上（`sparseCount_=512`）**行 184 的 640 分支实际不可达**
  （`maxS2Size_ > 512` 已提前返回；只有 `sparse_count > 640` 时该分支才可能生效）。
  因此设备上（A5，`sparse_count=512`）**唯一有效的翻转是 `maxS2Size_ > 512`，即 `S2 > 2048`。**

`supportFd_` 的 true / false 两条路径行为完全不同：

### supportFd_ = false（整行分配，无跨核 S2 切分）
- `AssignBlockToCore`（行 613–616）：`costLimit` 被抬到 `>= costInfo.maxS1GCost`，保证“整行”不会被切开；
- 跳过 `AssignByBlock` / `ForceAssign`（行 620–625）；
- 不记录 FD 头（行 631–635 均以 `supportFd_` 为前置条件）；
- `SplitFD` 不被调用（`BalanceSchedule` 行 719–721）；
- `GenMetadata` 里 `fdUsedVecNum = 0`，**所有 LD(AIV) 核 `LD_CORE_ENABLE=0`**（行 749–762）；
- 消费者侧 `quant_lightning_indexer_kernel.h` 中 `ldInfo.isLdCoreEnable == false`，`ProcessDecode()` 不调用 `ProcessLD()`；
- 结果：**每一行的整个有效 S2 由该行所属的那一个核用流式 top-k（`LiTopKVF` + `LiTopKGatherVF`）一次算出，直接写回输出。**

### supportFd_ = true（跨核 S2 切分 + LD 归约）
- 只要 `supportFd_ && numOfFdHead > 0`，就调用 `SplitFD`（行 674–699），把每个归约任务再按 load 拆到 AIV；
- `GenMetadata` 写 `LDMetadata[]`（行 748–762），`fdUsedVecNum > 0` → 部分 AIV 核 `LD_CORE_ENABLE=1`；
- 消费者侧 `ProcessDecode()` → `vectorService.ProcessLD()`（`quant_lightning_indexer_service_vector.h` 行 633–743）；
- 每个核只算自己 S2 子段的**局部 top-k（截断到 512 个候选）**，写入 workspace；
- LD 核把所有子段的局部候选拼接后调用 `topkOp_.LdTopK(...)`（`vf/quant_lightning_indexer_topk.h` 行 173–179），
  用 `LiTopKVF<true>` + `LiTopKLDGatherVF` 做**一次归约**，得到最终 top-k，再写回输出。

### 为什么两条路径会选出不同的索引
两者数学上都是 top-k，但 **等值（量化分数相同）时的取舍不同**：
- 非 FD：在行内按 S2 顺序做“分块 top-k → 归并”的流式归并（`LiTopKGatherVF`）；
- FD：各子段先各自截断到 512 个局部候选并重排，再按“核号顺序”拼接后一次性归并，且无效槽用
  `value=0 / index=-1` 填充（`ProcessLD` 行 664–665、699–707），这些 0 值会与真实 0 分候选竞争 K 边界；
- `LiTopKVF` 是基于 256 桶直方图的**值域** top-k，K 边界上的并列项按“进入向量的位置”截断，
  于是输入的**顺序/分块方式/填充**都会改变被选中的并列索引。

由此：
- M=1139（≈ `s2Size=285 < 512`）→ `supportFd_=false` → 路径 A；
- M=17523（≈ `s2Size=4381 > 512`）→ `supportFd_=true` → 路径 B；
两条不同算法 → 同一序列在不同 M 下选出不同 top-k 索引 → “indexer top-k 随 shape 变化”。

> 备注：任务里提到的 `:704` 与 `quant_lightning_indexer_kernel.h:668` 在 v9.1.0 中分别对应
> 消费者 kernel 的 `ProcessDecode()`（行 679–689）与其上游 `ProcessBaseBlock`/`ProcessTopK`；
> 真正的“跨核 top-k LD 归约”实现在 `arch35/quant_lightning_indexer_service_vector.h` 的 `ProcessLD()`。

### 两个同族怀疑点的核查结论（中性）
- `experimental/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/kv_quant_sparse_attn_sharedkv_metadata_aicpu.h:383`
  `bool supportFd = false;` —— v9.1.0 中**从未被赋值为 true**（全仓 `git grep "supportFd *="` 仅见 3 处声明、无赋值）。
  行 585/672/765/777 的读取恒为 false → **中性，非嫌疑点**。
- `experimental/attention/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_common.h:131`
  `bool isLDOpen = false;` —— v9.1.0 中**从未被赋值为 true**（全仓 `git grep "isLDOpen *="` 仅见 3 处声明）。
  attention 侧 `isLDOpen` 恒为 false → **中性，非嫌疑点**。

---

## (b) 具体改动

**文件**：`experimental/attention/quant_lightning_indexer_metadata/op_kernel_aicpu/quant_lightning_indexer_metadata_aicpu.cpp`

**改动点**：`CalcSplitInfo()` 末尾（v9.1.0 行 175–188），删掉三处 `maxS2Size_` 翻转，固定为单一常量。

### Before（v9.1.0，行 174–189）
```cpp
    }
    if (maxS2Size_ > sparseCount_) {                 // 175
        supportFd_ = true;                           // 176
        return;                                      // 177
    }                                                // 178
    validSocVersion_ = ProcessSocVersion();          // 179
    if (validSocVersion_ == ValidSocVersion::ASCEND910B) {      // 180
        if (maxS2Size_ > 2 * s2BaseSize_) {          // 181
            supportFd_ = true;                       // 182
        }                                            // 183
    } else if (validSocVersion_ == ValidSocVersion::ASCEND950) {// 184
        if (maxS2Size_ > 5 * s2BaseSize_) {          // 185
            supportFd_ = true;                       // 186
        }                                            // 187
    }                                                // 188
}                                                    // 189
```

### After（本分支，行 174–180）
```cpp
    }
    // 形状无关：FD 切分不能由 maxS2Size_（随 M/query 长度变化）决定。
    // supportFd_ 的两种路径（true=跨核 S2 切分 + LD 归约，false=整行分配）在等值
    // top-k 上可能选出不同索引，按形状翻转会使选中索引随形状变化。此处固定为单一
    // 常量，使小形状（M≈1e3）与大形状（M≈1.7e4）走完全相同的路径。
    supportFd_ = false;
}
```

> 也可不加任何赋值——头文件里 `bool supportFd_ = false;`（行 311）已是默认值。
> 这里显式写出是为了让“形状无关”成为代码里的明确契约。

### 为什么这样就能形状无关
把 `supportFd_` 固定为**单一常量**后，无论 `maxS2Size_`/M 多大，metadata 都进入**同一条**
分核与 top-k 路径（此处为 false → 整行分配）。选中索引只由该路径的单一算法决定，
不再出现“小 shape 走 A、大 shape 走 B”的分叉。

### 选哪个常量更安全
- **推荐 `false`（整行分配）**：
  - 它不做跨核 S2 切分、不做 workspace 归约、不引入 `0/-1` 填充，
    是“每行在自身完整有效 S2 上算一次 top-k”的最直接实现，最接近参考实现（golden）的语义；
  - 也是“小 shape / 短序列”本来就在走的路径；
  - 由 `AssignBlockToCore` 行 613–616 的 `costLimit` 抬升保证整行不被切开，行为确定。
- **不建议 `true`（恒开 FD）**：
  - 会把极短序列也强行切分 + 归约，既无收益，又把等值取舍交给“核号顺序 + 填充”的归约逻辑，
    与参考实现偏差更大；且多核/workspace 依赖更多。
- 若设备实测表明“参考结果”其实与 FD 归约一致，则把常量改为 `supportFd_ = true;` 即可（同样是单一常量、同样形状无关）。

### 依赖字段一致性（无需额外改动即自洽）
`supportFd_` 为 false 时，下列字段全部自然保持一致，**不需要改动其它文件**：
- `numOfFdHead` 恒为 0；`maxS2SplitNum` 恒为 0；
- `SplitFD` 永不调用；`fdUsedVecNum` 恒为 0 → 所有 `LDMetadata[*][LD_CORE_ENABLE_INDEX]=0`；
- `firstFdDataWorkspaceIdx[*]`（行 609）在 `preFdDataNum=0 / curKvSplitPart=1` 下恒为 0，消费者读到 `saveWorkSpaceIdx=0`，但 LD 核被禁用故不参与；
- `costLimit` 被抬到 `maxS1GCost`（行 613–616），保证整行分配成立；
- 因此该 .cpp **可独立编译通过**，消费者 kernel（`quant_lightning_indexer`）无需改动。

---

## (c) BUILD（构建 vendor 包，覆盖容器内 custom_transformer）

### 环境
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 保证 ASCEND_HOME_PATH / ASCEND_OPP_PATH 已设置
cd <ops-transformer 仓库根>                            # 切到 feat/indexer-supportfd-shape-invariant
```

### 关键 flag（已核对 v9.1.0 `build.sh`）
- `--experimental`：这两个算子在 `experimental/` 目录下，**必须**带上；
- `--soc=ascend950`：A5；
- `--ops=...`：只编译指定算子（同时 **`ENABLE_BUILT_IN=FALSE` + `ENABLE_BUILT_CUSTOM=TRUE`**，走 vendor 包路径）；
- `--vendor_name=<name>`：指定 vendor 名，默认 `custom`。构建产物落到
  `packages/vendors/${VENDOR_NAME}_transformer/...`，安装脚本 `vendor_name=${VENDOR_NAME}_transformer`
  （`cmake/custom_build.cmake:1175`），即 **`--vendor_name=custom` → `custom_transformer`**，正好覆盖容器内的 vendor。
- `--pkg`：生成 `.run` 安装包。

### 命令
```bash
bash build.sh --pkg --experimental --soc=ascend950 \
  --vendor_name=custom \
  --ops="quant_lightning_indexer,quant_lightning_indexer_metadata" -j16
```

> 说明：
> - 其中 **只有 `quant_lightning_indexer_metadata` 真正需要重建**（修复在它里面）；带上
>   `quant_lightning_indexer` 只是为了与既有使用方式一致、避免遗漏。若时间紧张可只编 metadata：
>   `--ops="quant_lightning_indexer_metadata"`。
> - AICPU 源码由 `op_kernel_aicpu/CMakeLists.txt` 经 `cmake/custom_build.cmake:277-278`
>   `add_subdirectory` 编译，**无需** `--aicpu_kernel`（该 flag 控制的是 device 侧 AICPU，不是这里的 custody AICPU op）。
> - 默认 `ENABLE_AICPU=TRUE`（`build.sh:39`），不要加 `--noaicpu`；不要加 `--opkernel`。

### 产物
```
build_out/cann-ops-transformer-custom_linux-<arch>.run   # arch = aarch64 / x86_64
```
（`cmake/custom_build.cmake:1213`：`cann-ops-transformer-${VENDOR_NAME}_linux-${ARCH}.run`；
确切文件名可 `ls build_out/*.run` 确认。）

---

## (d) INSTALL（安装到 custom_transformer，覆盖容器内旧文件）

vendor 安装脚本（`cmake/scripts/custom/install.sh`，包内 `vendor_name` 已被 sed 成 `custom_transformer`）：
- 目标目录优先级：`--install-path=<abs>` > `ASCEND_CUSTOM_OPP_PATH` > `ASCEND_OPP_PATH`；
- 落到 `$targetdir/vendors/custom_transformer/`，并更新 `$targetdir/vendors/config.ini` 的 `load_priority`，把
  `custom_transformer` 前置（`load_priority=custom_transformer,<others>`），从而**优先加载以覆盖内置/vendor 旧实现**；
- 安装是**逐文件覆盖/升级**，不会删除 vendor 目录内其它算子的文件（只更新本包所含文件）。

### 就地覆盖（推荐，最简单）
```bash
# 备份现场，便于回滚
cp -a "$ASCEND_OPP_PATH/vendors/custom_transformer" \
      "$ASCEND_OPP_PATH/vendors/custom_transformer.bak.$(date +%s)"

./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
# 如需显式指定安装目录：
# ./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path="$ASCEND_OPP_PATH"

# 重新 source vendor 环境
source "$ASCEND_OPP_PATH/vendors/custom_transformer/bin/set_env.bash"
```

### 作为新 vendor 安装 + 前置 ASCEND_CUSTOM_OPP_PATH（备选，避免动容器原件）
```bash
# 用不同的 vendor 名，例如 --vendor_name=customfix 构建后：
./build_out/cann-ops-transformer-customfix_linux-aarch64.run --quiet --install-path=/opt/customfix_opp
export ASCEND_CUSTOM_OPP_PATH=/opt/customfix_opp/vendors/customfix_transformer:$ASCEND_CUSTOM_OPP_PATH
# 注意：ASCEND_CUSTOM_OPP_PATH 若含 ":" 多路径，vendor 安装脚本会报错；
#       多路径时请用 --install-path 指定单一目录。
source /opt/customfix_opp/vendors/customfix_transformer/bin/set_env.bash
```
> `ASCEND_CUSTOM_OPP_PATH` 前置的目录会优先于 `$ASCEND_OPP_PATH`，因此新 vendor 可覆盖内置实现。

---

## (e) VERIFY

1. 重启相关进程/服务（确保重新加载 vendor so 与 AICPU kernel）。
2. 跑既有探针：
   - 成功判据：
     - `[CMPIDX-ALIGNED]` 在 P* 由 DIVERGENT 变为 **TOPK_SAME**；
     - `[LHID-POS] divergent=0`；
     - `curl#2..#5 == curl#1`（多次运行/多 shape 结果一致）。
3. 建议同时对比 M≈1e3 与 M≈1.7e4 两个 shape，确认两者选出的稀疏索引在相同前缀上一致
   （这正是本修复的目标：路径不再随 `maxS2Size_` 切换）。
4. 若探针未变绿：说明 root cause 不在此处或常量方向选反 → 见下方“下一步”。

---

## (f) ROLLBACK

### 设备侧（最快）
```bash
source "$ASCEND_OPP_PATH/vendors/custom_transformer/bin/set_env.bash" 2>/dev/null
rm -rf "$ASCEND_OPP_PATH/vendors/custom_transformer"
mv "$ASCEND_OPP_PATH/vendors/custom_transformer.bak.<ts>" "$ASCEND_OPP_PATH/vendors/custom_transformer"
# 或直接重装容器原始 custom_transformer 包：
# ./<原始>.run --quiet
```
重启进程使旧 vendor 生效。

### 代码侧
```bash
git checkout v9.1.0 -- experimental/attention/quant_lightning_indexer_metadata/op_kernel_aicpu/quant_lightning_indexer_metadata_aicpu.cpp
# 或 git revert <本修复 commit>
```
再按 (c)/(d) 重新构建并安装原始包。

---

## 若实测不符：下一步候选

本修复是 **CANDIDATE**。若设备验证后 `[CMPIDX-ALIGNED]`/`[LHID-POS]` 无改善，则
`supportFd_` 并非真正的形状依赖根因（或方向相反），下一批候选：
1. **反向常量**：改为 `supportFd_ = true;`（同样形状无关），确认参考实现究竟对应哪条路径；
2. 消费者侧 `quant_lightning_indexer_service_vector.h` `ProcessTopK`/`ProcessLD` 中的
   `curS2StartIdx` 偏移与 `-1/0` 填充（行 428/471/562/588 附近）与 `topkCountAlign256_` 的边界处理；
3. `vf/quant_lightning_indexer_topk.h` 的 `LiTopKVF` / `LiTopKGatherVF` / `LiTopKLDGatherVF`
   在 K 边界并列时的截断取舍（真正的“等值 tie-break”发生地）；
4. metadata 的 `trunkLen`(16384) / `topkCountAlign16_` 对齐与 `saveWorkSpaceIdx` 的 workspace 布局。
