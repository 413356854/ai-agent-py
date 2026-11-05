# DSH（deepseek-harness）摘要压缩方案提取

> 提取对象：`C:\Users\ljf\Desktop\dsh\deepseek-harness`
> 目的：给 `ai-agent-py`「通用 Agent」的上下文压缩提供可直接借鉴的工程方案。
> 范围：`packages/compaction/*`、`packages/llm/token-meter`、`docs/subsystems/compaction.md`。
> 说明：本文只做提取与分析，不含对 `ai-agent-py` 的代码改动。

## 1. 一句话概括

DSH 把「长对话压缩」做成一组可组合的能力：**先修剪（工具输出、图片），再按 token 压力把最老的一段历史替换成结构化摘要**。替换不是删除，而是通过 **append-only 会话日志 + surface（当前可见面）投影** 完成：原文永久保留、可回放，模型只看到「检查点摘要 + 最近保留的原文」。摘要调用复用原请求的系统提示词、工具与历史前缀，尽量命中 provider 的 KV cache；此外还有手动 `/compact`、上下文溢出重试、模型切换前预检等配套机制。

核心组成：

| 层 | 组件 | 作用 |
|---|---|---|
| 契约 | `@deepseek-ai/dsh-compaction` | 压缩操作、摘要格式、工具配对边界、检查点来源标记 |
| 后端 | `dsh-compaction-basic` | 阈值、保留范围选择、摘要调用、事务与重试 |
| 预处理 | `dsh-compaction-tool-result-pruner` | 先修剪过大的工具输出（无模型调用） |
| 预处理 | `dsh-compaction-image-offload` | 图片超预算时替换为占位文本 |
| 手动入口 | `dsh-command-compact` | `/compact` 命令，空闲时按需压缩 |
| 计量 | `dsh-token-meter` | 基于日志回放的 token / 压力估算，所有压缩决策共用 |

## 2. 模块地图（源码位置）

| 文件 | 职责 |
|---|---|
| `packages/compaction/README.md` | 能力族总览、包关系 |
| `docs/subsystems/compaction.md` | 子系统契约：事件、`CompactionResult`、服务接口 |
| `packages/compaction/compaction/src/types.ts` | `compaction/*` 事件、`CompactionResult` |
| `packages/compaction/compaction/src/checkpoint.ts` | 检查点消息来源 `compact-checkpoint` 及识别函数 |
| `packages/compaction/compaction/src/tool-pairing.ts` | 工具调用/结果配对的「安全切口」 |
| `packages/compaction/compaction-basic/src/config.ts` | 阈值/保留策略解析、校验、按模型覆盖 |
| `packages/compaction/compaction-basic/src/region.ts` | 范围选择、事务、稳定性校验、替换落盘 |
| `packages/compaction/compaction-basic/src/summarizer.ts` | 摘要调用、prompt、检查点 framing |
| `packages/compaction/compaction-basic/src/index.ts` | 监听器：pre-step 压力、模型切换、溢出恢复、`/compact` |
| `packages/compaction/compaction-tool-result-pruner/src/index.ts` | 工具结果 head/middle/tail 修剪 |
| `packages/compaction/compaction-image-offload/src/index.ts` | 图片卸载 |
| `packages/llm/token-meter/src/estimate.ts` | 固定 token 估算（4 字符/token + 结构开销） |

## 3. 触发条件与阈值

### 3.1 阈值公式

设：

- `W`：当前路由模型的上下文窗口（adapter 声明）
- `O`：本次请求预留的输出 token（`maxTokens`，否则 adapter 默认，否则 0）
- `B`：额外的压力余量 `headroomTokens`
- `R`：保留的最近历史预算（`retainRatio` 或 `retainTokens`）

则：

```text
messageBudget   = W − O
pressureBudget  = messageBudget − B
thresholdTokens = floor(min(W × thresholdRatio, pressureBudget))
retainTokens    = floor(messageBudget × retainRatio)   # 或显式 retainTokens
```

默认值（`compaction-basic`）：

| 配置 | 默认 | 含义 |
|---|---|---|
| `thresholdRatio` | `0.8` | 触发阈值占窗口比例 |
| `headroomTokens` | `65536` | 输出预留之外再留的压力余量 |
| `retainRatio` | `0.16` | 保留最近历史的预算比例（与 `retainTokens` 二选一） |
| `retainTokens` | — | 保留最近历史的绝对 token 数 |
| `maxTokens` | `headroomTokens` | 摘要调用的输出上限 |
| `compactionRetries` | `1` | 首次压缩后仍超阈值时的额外尝试次数 |
| `maxOverflowRetries` | `1` | 确认上下文溢出后的最大重试次数 |
| `modelPolicies` | `[]` | 按 `provider/model` 精确覆盖 |
| `auto` | `true` | 是否启用自动压缩与溢出恢复 |

校验约束（加载时失败）：

- 未知配置键、重复的 model policy、非法 token 数直接拒绝；
- `retainRatio` 与 `retainTokens` 互斥；
- 保留预算必须小于触发阈值；
- `W − O − B` 必须为正；
- 小窗口模型必须显式配 `headroomTokens`（默认 64K 放不进小窗口）。

### 3.2 四个触发入口

| 入口 | 时机 | 行为 |
|---|---|---|
| 自动压力 | `agent/pre-step`，每次请求推导前 | 用最近一次路由请求的用量估算压力；超过 `thresholdTokens` 时修剪 → 摘要 |
| 模型切换 | `agent/request`，目标 provider/model 变更前 | 用**目标模型**的上下文/输出/阈值评估；不满足则用**旧模型**摘要（显式配置的 summary 目标优先）；失败拒绝切换（`MODEL_SWITCH_COMPACTION_FAILED`） |
| 上下文溢出 | `agent/request-error`，provider 返回 `CONTEXT_WINDOW_EXCEEDED` | 绕过普通阈值与保留策略，先 prune，再选「最大可压缩的头部」，只保留最少尾部；成功后重试一次 |
| 手动 | `/compact` 命令 | 空闲 agent 上按 `compactNow()` 压缩一个有用范围，即使未到阈值 |

`compaction-basic` 在每次自动压缩后都会重新计量；若修剪已把压力降到阈值以下，就**跳过摘要**，只留下修剪结果。

## 4. 压缩范围如何选

`selectCompactableRange()`（`region.ts`）的核心规则：

1. **永远不动 surface 第 0 个 `system/message`**（当前系统提示词）；范围从第一个非 system 节点开始。
2. 从尾部向前累加每个节点的 token，直到累计 ≥ `retainTokens`，得到保留尾部起点 `keepFromIdx`。
3. 从 `keepFromIdx` 继续向前调整，直到切口 `toolPairingBalancedBefore()` 为真：保证不会把一次 `assistant tool_call` 和它的 `tool/result` 拆开。
4. 范围 = `[第一个非 system 节点, keepFromIdx − 1]`；没有安全范围则返回 `null`。
5. 自动压力路径要求整个 surface 在选择/摘要期间不变；手动 `/compact` 只要求被选中的 span 不变（期间注入的其他上下文可保留）。

工具配对判定（`tool-pairing.ts`）是增量折叠：

```text
assistant/message 中 tool-call 数量 → inProgress + N
tool/result                        → inProgress − 1
inProgress === 0 的切口才算 balanced
```

这样压缩后消息序列对 provider 仍然合法。

## 5. 摘要生成

### 5.1 用哪个模型

优先级（`summarizer.ts`）：

1. 显式配置的 `summarizationProvider` + `summarizationModel`；
2. 最近一次已路由请求的 provider/model；
3. `AgentOptions` 的 provider/model；
4. 都没有则报错，提示配置摘要目标或先路由一次请求。

摘要调用是一次独立的 `ctx.llm.stream()`，`purpose = 'compaction'`，`maxTokens = 配置的 maxTokens`，只取返回文本；图片输出直接报错（`UNSUPPORTED_CONTENT`），空摘要报错，达到 `maxTokens` 截断也报错（不允许不完整检查点）。

### 5.2 摘要请求长什么样（KV cache 复用）

摘要请求不是「另起一个摘要系统提示词」，而是按原请求前缀重放：

```text
[原系统提示词]
[原请求的工具 schema]
[被压缩范围内的历史消息（按 surface 顺序，包括历史中的 system/message）]
[最后一条 user 消息：压缩指令]
```

这样它与最近一次真实请求共享尽可能长的前缀，provider 的 warm prefix cache 可以复用，只有末尾指令与摘要输出是新的 token。代价是摘要输入 = 系统提示 + 工具 schema + **被压缩范围**（不含保留的尾部），范围越大这次辅助请求越贵。

### 5.3 压缩指令（原文，节选完整结构）

```markdown
You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.

Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.

## Primary Request and Intent
- [the user's original and evolving goals; quote verbatim where the exact wording matters]

## Key Technical Concepts
- [technologies, frameworks, patterns, and conventions in play]

## Files and Code
- [exact path: why it matters, key changes or snippets]

## Errors and Fixes
- [error: how it was resolved, plus any related user feedback]

## Pending Jobs
- [explicitly requested work not yet completed]

## Current Work
- [precisely what was in progress at this checkpoint]

## Next Step
- [the single next action, directly in line with the most recent request, or "(none)"]

## Critical Context
- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]

Rules:
- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.
- Capture user feedback and explicit instructions faithfully, especially corrections.
- Do NOT mention this summarization request or that the context was compacted.
- Output only the checkpoint text: do not call any tool or take any other action.
- If the conversation already contains a <compacted-summary> block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure.
```

设计要点：

- 8 个固定章节，不允许缺省；空章节写 `(none)`；
- 强调保留精确路径、命令、错误串、标识符、数值、函数签名；
- 要求合并旧检查点、丢弃过期事实，而不是把旧摘要原样复制；
- 要求不提及「压缩」本身，避免模型在后续回答里泄露实现细节。

### 5.4 摘要必须真的变小

生成后会用 token meter 估算 framed 检查点（含 preamble 与标签）的 token 数，并与被替换 span 的**路由计价** token 比较：

```text
framedSummaryTokens < shadowedRouteTokens   # 否则抛错，不提交替换
```

这是防止「摘要比原文还长」的硬约束。若压不下来，自动路径会记录失败并继续当前轮（不阻塞对话）；模型切换/手动路径则按错误处理。

## 6. 替换、事务与持久化

### 6.1 事件序列

压缩不是删除历史，而是在 append-only 会话日志中追加：

| 顺序 | 事件 | 作用 |
|---|---|---|
| 1 | `compaction/start` | 获取压缩锁；`turn` 为自动轮次号，或 `null` 表示独立的 `/compact` |
| 2 | `compaction/summary` | 记录摘要、被替换范围、被遮蔽节点 seq、token 估算、摘要模型与用量等（log-only） |
| 3 | `user/message`（带 `surfaceOp: { op: 'replace', startSeq, endSeq }`） | 唯一会改变 surface 的事件：用一条检查点消息替换旧范围 |
| 4 | `compaction/end` | 释放锁；`error` 记录失败 |

原始消息仍在日志里，只是被 surface 投影遮蔽；回放/审计/恢复都能拿到完整原文。`compaction/start` 没有配对 `compaction/end` 即视为**遗留锁**，会阻塞后续压缩；但若它早于新的 `session/end-seed` 边界，则视为上一生命周期残留，不阻塞。

### 6.2 替换消息的 framing

```markdown
This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.

<compacted-summary>
...摘要正文...
</compacted-summary>
```

- 用 **user 角色**消息承载摘要（不是 system），让后续模型把它当作「已确立的背景」；
- 消息 source 带 `compact-checkpoint` 标记与 `compactionId`，客户端/服务端可识别并关联；
- 检查点替换范围，最近的保留尾部原样跟在后面；
- 替换会让 KV cache 从第一个被替换的 token 起失效；替换范围之前的前缀仍可复用。

### 6.3 事务与取消

- 选择、校验是只读的；`compaction/start` 与异步摘要之间没有让步点，因此锁是可靠的；
- 摘要完成后再校验 surface/span 是否仍稳定（自动：整个 surface；手动：选中 span）；
- 任何失败都会尝试一次 `compaction/end`；close 本身失败会故意留下可检测的 orphan；
- 手动压缩要求 agent 空闲；期间到达的 prompt 会排队，压缩结束（或持久化 checkpoint 完成）后再开始；
- 取消信号贯穿摘要调用；取消仍以调用方 reason 为准。

## 7. Token 计量

`dsh-token-meter` 是所有压缩决策的唯一计价来源：

- 文本：`ceil(字符数 / 4)`（固定启发式），每个 block 加 4 token 结构开销，每条消息加 4 token role 开销；工具 schema 按 JSON 字符串计；
- 图片：路由声明了图片计价时用 provider 的视觉 token，否则固定启发式；
- provider 返回的 usage 仅在**请求 envelope 完全一致**时复用；否则整体重新估算；
- 提供 `totalTokens`（请求+响应压力）、`surfaceTokens`、逐节点 `tokens` / `heuristicTokens`；
- **明确承认局限**：4 字符/token 会严重低估 CJK 文本和 JSON Schema；精确 tokenizer 尚未落地，是 Dev Note 里明确记录的技术债。

这意味着 DSH 的阈值是用「保守但统一」的估算来触发的；在中文场景直接照搬 4 字符/token 会低估约 2 倍以上，必须换成更贴近目标模型的估算。

## 8. 工具输出修剪（pruner）

- 只在一个压缩触发已经成立后运行（压力或溢出），**未到压力不动历史**；
- 对每个超预算工具结果：保留 head + 中间标记 + tail，默认：

| 配置 | 默认 |
|---|---|
| `thresholdChars` | `8192` |
| `headChars` | `4096` |
| `tailChars` | `1024` |

- 替换文本形如：`...\n\n[... tool result middle pruned ...]\n\n...`；
- 按 Unicode code point 切，不会切断 UTF-16 surrogate pair；富内容 block（图片等）保持顺序；
- 只改文本，工具调用、错误、metadata 保留；
- 无模型调用，可能直接把压力降到阈值以下，从而**跳过摘要**；
- 原始结果仍在 append-only 日志里；替换消息通过 `sourceEventSeqs` 指向原文，可回放；
- 局限：字符预算 ≠ token 预算；只按头尾切，不理解中间语义。

## 9. 图片卸载（image offload）

- 仅在 provider 返回 `IMAGE_OFFLOAD_REQUIRED`（图片超出路由预算）时触发；
- 按请求顺序选择最老的、尚未卸载的图片，替换为「附件名 + 只读路径」的占位文本；
- 选择是持久的（`image/offload` 事件），后续请求继续生效；卸载后不会自动恢复；
- 不消耗常规 provider retry 预算；摘要过程中失败时也走 `compaction/summary-error` 恢复；
- 与文本压缩的关系：图片卸载独立于摘要，但同样通过 surface 投影影响模型输入。

## 10. 手动 `/compact`

- 仅空闲 agent 可用；正在 turn 中或压缩中时返回稳定错误文案；
- 不带参数；压缩一个「有用的最老平衡范围」，可低于自动阈值；
- 返回压缩了多少历史项、节省的估算 token；
- 压缩期间提交的 prompt 进入普通 FIFO，等压缩完成（含持久化 checkpoint）后再开始；
- 命令生命周期 `command/run` / `command/done` 是 log-only，不进入模型历史；
- 失败分类清晰：`busy` / `cancelled` / `changed` / `summary` / `commit` / `persistence`。

## 11. 失败语义与已知边界

| 场景 | 行为 |
|---|---|
| pre-step 自动压缩失败 | 记录 warning，继续当前 turn（不因压缩失败中断对话） |
| 模型切换压缩失败 | 拒绝切换，保留旧模型；报 `MODEL_SWITCH_COMPACTION_FAILED` |
| provider 确认上下文溢出 | 尝试一次「先 prune + 最大头部压缩」，只有 surface 的 replacement generation 前进才授权重试 |
| 摘要失败/空/截断/比原文长 | 记录失败，不提交替换；手动/模型切换路径报错 |
| 修剪中途失败 | 已提交的 trim 保留，整次运行失败 |
| 无法压缩 | 系统提示词、工具 schema、请求前缀不可压缩；单个不可分割的大单元（如一个巨大 tool call）无法被切分 |
| 估算误差 | 4 字符/token 低估 CJK 与 JSON Schema |
| 图片跨模型 | 不同图片价格可能让摘要缩小校验失败 |

## 12. 可借鉴到 `ai-agent-py` 的映射

| DSH 机制 | 我们的现状 | 建议 |
|---|---|---|
| 阈值公式 `min(W×ratio, W−O−B)` | 改造前只有消息数窗口（本次已改为 token 预算） | 引入 token 预算；用模型窗口、输出预留、headroom 计算；中文估算用 1.5~2 字符/token，或 tiktoken |
| 每次请求前 pre-step 检查 | 每轮 API 入口组装历史 | 在 `/chat` 与 `/chat/stream` 前统一做预算/摘要检查 |
| 工具输出先修剪 | 工具输出全量进 `ToolMessage`；20k 只作用于落库/SSE | 在工具节点或提示词组装处做 head+marker+tail 修剪；完整结果放 `artifact`/MySQL |
| 只修剪到阈值以下就跳过摘要 | 无 | 修剪后重新计数，未超阈值则不调用摘要模型，省钱 |
| 结构化摘要 prompt（8 节） | 无 | 中文改写为 8 节；保留路径、命令、错误、数值、未完成事项；合并旧摘要 |
| `<compacted-summary>` + preamble | 无 | 用 user 角色检查点消息承载摘要，MySQL 存 summary + `until_seq` |
| 摘要必须缩小 | 无 | 生成后估算 framed 摘要 vs 被替换历史，不缩小则拒绝提交 |
| 手动 `/compact` | 无 | 可做 `POST /sessions/{id}/compact` 或前端按钮 |
| 溢出后重试一次 | 无 | 捕获 OpenAI 兼容接口的 context-length 错误，强制压缩后重试一次 |
| 模型切换预检 | 无（当前单模型） | 将来支持多模型时再加 |
| append-only + surface 投影 | MySQL 消息表 + 前端按序渲染 | 消息不删；新增 summary 表与压缩标记；前端可灰显/折叠已压缩消息 |
| KV cache 复用 | 每轮重建 messages | 摘要调用可复用系统提示词+工具+历史前缀（可选优化） |
| 4 字符/token | 未计量 | **不要照搬**；中文至少按 2 字符/token，或按模型 tokenizer |
| 64K headroom | 无 | 我们的应用规模建议 headroom 4K~8K，按模型窗口配置 |
| 图片卸载 | 暂不支持图片输入 | 暂不需要 |

建议的落地顺序（与之前方案对齐）：

1. **L1 工具输出修剪**：单次/单轮预算，head+marker+tail，完整结果保留 artifact/MySQL；
2. **L2 token 预算选择历史**：从最新往前保留「最近完整轮 + 预算内历史」，超出部分交给摘要；
3. **L3 滚动摘要**：新增 `tb_agent_chat_summary`，结构化摘要 prompt，替换/前插检查点，校验缩小；
4. **L4 溢出重试 + 手动压缩**：捕获 context-length 错误、压缩后重试一次；提供手动入口。

不建议照搬的部分：

- 完整的 append-only 事件日志 + surface 投影（我们的 MySQL 消息表 + 前端已有顺序渲染，改造成本高；用 summary 表 + `until_seq` 即可）；
- 64K headroom、65536 的摘要输出上限（针对超长 coding session，我们的场景小得多）；
- 图片卸载、模型切换预检（当前没有对应需求）；
- 多套 `compaction/*` 事件与事务锁（单实例 MySQL 场景可用更简单的行锁/条件更新替代）。

## 13. 与 LangChain 官方方案的关系

| 能力 | LangChain 官方 | DSH |
|---|---|---|
| 摘要触发 | `SummarizationMiddleware(trigger, keep)` | 阈值公式 + pre-step / overflow / model-switch / manual |
| 工具输出清理 | `ContextEditingMiddleware` + `ClearToolUsesEdit` | 独立 pruner，head/middle/tail，触发后先跑 |
| 摘要 prompt | 默认 `DEFAULT_SUMMARY_PROMPT`（SESSION INTENT 等） | 8 节结构化 prompt，强调路径/命令/错误/下一步 |
| 持久化 | checkpointer（Postgres/SQLite/内存） | append-only session log + surface 投影 |
| 失败恢复 | 摘要重试、上下文溢出已处理 | 事务锁、orphan 检测、溢出重试、模型切换保护 |
| 中文支持 | 默认估算字符/token 需调整 | 同样承认 4 字符/token 低估 CJK |

结论：DSH 的方案在 LangChain 官方中间件的基础上，补齐了**事务、可回放替换、按路由计价、溢出/模型切换/手动入口**这些工程细节。我们可以借它的「语义与参数」，不一定借它的「事件日志架构」。

## 14. 关键默认值速查

```text
thresholdRatio       = 0.8
headroomTokens       = 65536
retainRatio          = 0.16
maxTokens            = headroomTokens (65536)
compactionRetries    = 1
maxOverflowRetries   = 1
auto                 = true
pruner.thresholdChars= 8192
pruner.headChars     = 4096
pruner.tailChars     = 1024
token heuristic      = 4 chars/token + 4 token/block + 4 token/message
```

## 15. 主要来源

- `packages/compaction/README.md`
- `packages/compaction/compaction-basic/README.md`
- `packages/compaction/compaction-basic/src/config.ts`
- `packages/compaction/compaction-basic/src/region.ts`
- `packages/compaction/compaction-basic/src/summarizer.ts`
- `packages/compaction/compaction-basic/src/index.ts`
- `packages/compaction/compaction/src/types.ts`
- `packages/compaction/compaction/src/checkpoint.ts`
- `packages/compaction/compaction/src/tool-pairing.ts`
- `packages/compaction/compaction-tool-result-pruner/README.md`
- `packages/compaction/command-compact/README.md`
- `packages/llm/token-meter/README.md`
- `packages/llm/token-meter/src/estimate.ts`
- `docs/subsystems/compaction.md`
- `packages/compaction/compaction-image-offload/README.md`
