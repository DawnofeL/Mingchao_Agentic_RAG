# Orchestrator 题库生成 Skill —— 设计思路

本文档完整记录该 skill 的设计动机、关键取舍和不做的事，目的是让未来维护者（或我自己）能立刻理解 why，而不是只看到 what。

---

## 1. 为什么需要这个 skill

`mingchao_people_evaluation` 和 `mingchao_timeline_evaluation` 两个 skill 各自覆盖了单域检索能力：

- people skill：测人物结构化检索（people_search / relationships_search）
- timeline skill：测事件结构化检索（event_search）

但 Agentic RAG 的核心能力是 **Orchestrator 多任务编排**——README Type 5 描述的子任务拆分、依赖图调度、扇出枚举、阻塞传播。这层能力**两个单域 skill 都测不到**：

- 一道 single 题永远不会进 Orchestrator
- 一道纯人物或纯事件的 subtasks 题只考拆分，不考跨域调度

Orchestrator skill 的目标就是补齐这道缺口：**生成的每道题必须强制跨 people + timeline 两个域，且必有 ≥ 2 个 task 形成依赖图**。

---

## 2. 核心原则：跨域强制

这是整个 skill 的灵魂约束。

**定义**：一道题被认定为 orchestrator 题，当且仅当 `expected_tasks` 中的 `intention` 同时包含 `people` 和 `timeline`。

**为什么如此严格**：

- 若不强制跨域，agent 会大量生成单域 subtasks 题（QU skill 路由出来的 multi_enum），稀释 orchestrator 信号
- 跨域是 orchestrator 区别于其他 skill 的**唯一身份特征**，必须硬约束
- 单域 subtasks 题该归 people / timeline skill 出，不属于 orchestrator 范畴

**判别工具**：self_check Part 1 第一项检查就是它，违反直接 `⚠️`。

---

## 3. 子类型设计：为什么是 4 种

README Type 5 列了 5 种子任务结构（5a-5e）。我刻意去掉 5d（异类并行：文臣 vs 武将），保留 4 种：

| 子类型 | README 对应 | 拓扑 |
|---|---|---|
| 并行跨域 | 5a | 全部 depends_on=[] |
| 顺序两级 | 5b | t1 → t2 |
| 扇出枚举 | 5c | t1 → t2, t3, ...（多个扇出） |
| 严格链式 | 5e | t1 → t2 → t3 → ... |

**为什么去掉 5d**：5d"异类并行"的典型例子是"文臣和武将"——两个 task 都查 people，**本质是同域并行**。把它纳入 orchestrator skill 会违反"跨域强制"原则。同域并行属于 people skill 的 multi_enum 范畴。

**为什么不再细分**：sub_type 的粒度只需到"任务图拓扑"层即可，再往下细分（如按 query_kind、按域分布）会产生大量正交组合，每个组合的题量稀薄，自检也难写。这层粒度恰好对应 README Type 5 的命名，前后一致。

---

## 4. 为什么用 era 而不是 id 区间作锚点

`people_evaluation` 用 people_id 区间，`timeline_evaluation` 用 event_id 区间。orchestrator skill 故意不沿用这个套路。

**理由**：

1. **id 区间无法保证跨域交叠**。people_id 1-30 是按拼音排序的某 30 个人，timeline_id 1-30 是某 30 个事件，两者人物-事件交集稀薄，难以出"涉及该人物的事件"这类跨域题。
2. **era 是 orchestrator 题目天然的时段锚点**。Type 5 例题里几乎所有问题都按时段提问（"永乐年间……"、"洪武四大案"、"靖难之役期间……"）。
3. **era 直接保证两个域有数据交叠**。该 era 的人物天然涉及该 era 的事件，反之亦然。

**实际数据验证**：
- 永乐 era：23 人物 + 31 事件
- 洪武 era：39 人物 + 41 事件

足以支撑 4 个子类型 × 2 道 = 8 道题的生成。

**多 era 支持**：参数允许 `--era "建文,永乐"`，覆盖跨年号题（如靖难之役全程）。

---

## 5. 关键创新一：`expected_tasks` 字段

人物 / timeline 题库的格式是 `{question, keypoints[]}`。orchestrator 题库增加 `expected_tasks` 字段。

**为什么**：

- orchestrator 题的"正确答案"不只是事实，还包括**任务图拓扑**——QU 应该如何拆分这道题
- 没有 expected_tasks，self_check 无法验证：跨域强制是否满足？依赖图是否符合 sub_type？keypoint 域是否匹配 task intention？
- 写出 expected_tasks 也强迫出题人想清楚拆分逻辑，避免出"看起来跨域但其实拆不出"的伪 orchestrator 题

**self_check 用 expected_tasks 做什么**：

- 验证跨域强制（Part 1 检查 1）
- 验证 task_id 一致性（Part 1 检查 2）
- 验证 depends_on 合法性（Part 1 检查 3）
- 验证 sub_type 拓扑合规（Part 1 检查 4）
- 验证 source-intention 匹配（Part 1 检查 5）

**self_check 不用 expected_tasks 做什么**：

- **不**与 QU 实际输出做字符串/字段比对。QU 可能产出多种合理拆分，硬比对会导致大量假阳性
- expected_tasks 是"出题人对 QU 期望"的声明，不是 ground truth

---

## 6. 关键创新二：每条 keypoint 标 `task_id`

人物 / timeline 题库的 keypoint 没有归属，因为题目本身就是单 task。orchestrator 题目是多 task，每条 keypoint 必须明示它支撑哪个 task。

**为什么**：

- 验证"每个 task 至少 1 个 keypoint"——否则某个 task 是空话，没法评测
- 验证"source 域与 task intention 匹配"——一道并行跨域题里，标了 `intention=people` 的 task 对应 keypoint 必须来自 people.json，不能拿 timeline 的字段冒充
- 评测 LLM 答案时可以分 task 计算覆盖率：t1 命中几条、t2 命中几条，更细粒度

**反例（为什么必须有 task_id）**：

```json
{
  "expected_tasks": [
    {"task_id": "t1", "intention": "timeline"},
    {"task_id": "t2", "intention": "people"}
  ],
  "keypoints": [
    {"answer": "迁都北京", "source": "timeline[XXX].event"},
    {"answer": "姚广孝",   "source": "people[姚广孝].roles"}
  ]
}
```

没有 task_id，self_check 无从判断"姚广孝"是支撑 t1 还是 t2，也无法验证 source 域是否匹配 task intention。

---

## 7. self_check 的三部分结构

| Part | 检查项 | 失败硬度 |
|---|---|---|
| Part 1 | 结构性硬约束（跨域 + task 图 + source/intention） | 硬错误，必须 0 |
| Part 2 | keypoint 溯源（answer 能从 source 字段读出） | 硬错误，必须 0 |
| Part 3 | 问题模板多样性（疑问词分布） | 软警告，控制在 70% 以下 |

**为什么把 Part 1 单独提到最前**：人物 / timeline skill 的 self_check 只有溯源 + 多样性两部分。orchestrator 新增的"跨域 + 任务图"约束属于结构性硬错误，违反它意味着题目根本不是 orchestrator 题，必须先过这一关再谈 keypoint 是否对得上。

**为什么沿用 Part 2 / Part 3**：和单域 skill 保持评测口径一致，可以横向对比"orchestrator 题库的多样性是不是比单域题库差"。

---

## 8. 我刻意不做的事（避免过拟合的清单）

写这种生成器很容易陷入"功能越多越好"的陷阱。以下是被反复评估后**主动砍掉**的功能：

| 不做 | 理由 |
|---|---|
| 出 5d 异类并行题 | 同域并行属 people skill 范畴 |
| 出阻塞传播专项题 | 需要"知识库不在的内容"作上游，会污染溯源 |
| 嵌入 LLM-as-judge 评分 | 评测口径要和现有两个 skill 一致，靠 keypoint 命中率衡量 |
| 与 QU 输出做严格字符串比对 | QU 拆分有多种合理答案，硬比对假阳性高 |
| 自动检测 final_answer 合成质量 | 那是另一个评测维度，混进来失焦 |
| 支持 era 之外的锚点（如 participants 锚点） | era 已经能覆盖典型 orchestrator 题，加锚点会让参数空间爆炸 |
| 默认 n 超过 2 | 单 era × 4 子类型 × n=2 = 8 题已足够压力测试，更多容易稀释每个子类型的代表性 |

---

## 9. 文件结构

```
mingchao_orchestrator_evaluation/
├── SKILL.md                  # agent 入口，生成流程 + 自检指引
├── README.md                 # 本文档，设计思路
├── orchestrator_loader.py    # 按 era 加载人物 + 事件跨域视图
└── self_check.py             # 三部分自检：结构 / 溯源 / 多样性
```

**与 people / timeline skill 的差异**：

- 多一个 README.md（本文档）。单域 skill 设计简单不需要单独文档，orchestrator 涉及跨域强制、expected_tasks、keypoint task_id 等关键决策，必须记录
- loader 是新写的（按 era 而非 id 区间），不复用现有两个 loader
- self_check 多了 Part 1 结构性约束，前置于溯源检查

---

## 10. 与上游模块的关系

```
            ┌─ rag/agent/skills/query_understanding.md  (QU 拆分规则)
            │
本 skill ───┼─ rag/graph/nodes/orchestrator.py         (依赖图调度器)
            │
            └─ rag/agent/skills/final_answer.md        (终答合成)
```

本 skill 生成的题库用于评测以上三个模块的协同。具体来说：

- 题目是否被 QU 正确拆分 → QU 准确率
- 拆出的子任务是否被 Orchestrator 正确并发/串行执行 → 调度器正确性
- 多个子任务结果是否被 final_answer 完整合成 → 合成器召回率

跨域强制确保题目必然触发 Orchestrator 分支，从而真正考验调度器（而不是被 single 直通绕过）。
