---
name: mingchao_orchestrator_evaluation
description: 输入 era（年号锚点）和每子类型题数 n，生成用于 Orchestrator 多任务编排压力测试的评测题库 JSON。每道题必须同时涉及 people 和 timeline 两个域，覆盖并行跨域、顺序两级、扇出枚举、严格链式四种任务图结构。
---

# Orchestrator 跨域多任务题库生成器

## 参数

| 参数 | 是否必填 | 含义 | 默认 / 示例 |
|---|---|---|---|
| `era` | 必填 | 年号锚点，单值或逗号分隔多值 | `"永乐"` / `"建文,永乐"` |
| `people_json` | 可选 | 人物 JSON 路径 | 默认：`…/data/people_timeline/mingchao_people.json` |
| `timeline_json` | 可选 | 事件 JSON 路径 | 默认：`…/data/people_timeline/mingchao_timeline.json` |
| `n` | 可选 | 每子类型题数 | 默认 `2` |

---

## Part 1：生成题库

### Step 1：加载跨域视图

严禁直接读取整个 people / timeline JSON（人物 185 条 + 事件 258 条，一次性读会越界）。唯一方式：

```bash
# 使用项目默认路径
python3 {SKILL_DIR}/orchestrator_loader.py --era {era}

# 使用自定义路径
python3 {SKILL_DIR}/orchestrator_loader.py --era {era} --people_json {people_json} --timeline_json {timeline_json}
```

输出两段：先列该 era 内全部人物，再列该 era 内全部事件。标记每位人物的 people_id、name、roles、relationships，以及每条事件的 event_id、year、era、event、participants、outcome。

---

### Step 2：按子类型出题

每个子类型出 `n` 道题。**每道题必须满足跨域强制**：`expected_tasks` 中的 `intention` 必须同时包含 `people` 和 `timeline`。

**四种子类型的拓扑与出题策略**：

| 子类型 | 拓扑 |
|---|---|
| **并行跨域** | 多个 task 全部 `depends_on=[]` |
| **顺序两级** | 恰好 2 个 task，t2 `depends_on=[t1]` |
| **扇出枚举** | 1 个 t1 + 多个 task 都 `depends_on=[t1]` |
| **严格链式** | t1 → t2 → t3 → ...，每个 task 依赖前一个 |

**并行跨域**：选一位**该 era 活跃的核心人物**或**一个跨域议题**，并行查其在 timeline 中的参与事件、与其他人物的关系、相关分析。

- 合格：`"于谦参与了哪些重大历史事件、于谦与哪些人有决定性关联、书中如何评价他？"`
  - t1 timeline（参与事件，multi_enum），t2 people（关键关系人，multi_enum），t3 people（评价，analysis）
  - 三 task 都 `depends_on=[]`
- 不合格：`"于谦参与了哪些战役、又有哪些重要的胜负？"` — 两 task 都落在 timeline，没跨 people

**顺序两级**：t1 timeline 拿事件命名集合，t2 people 查这些事件涉及的人物。两 task 都 multi_enum。

- 合格：`"永乐年间最重要的几件大事分别是什么时候发生的、各自涉及哪些核心人物？"`
  - t1 timeline（大事 + 时间），t2 people（涉及人物，depends_on=[t1]）
- 不合格：t1 拿人物集合再 t2 查事件 — t1 应该是 timeline（事件锚点），t2 才是 people

**扇出枚举**：t1 拿小集合（建议 3-5 项），t2、t3 各自对集合每项查不同维度，**至少一条扇出走 people、至少一条走 timeline**。

- 合格：`"洪武四大案分别是什么、各自牵连了哪些重要人物、各自造成了什么后果？"`
  - t1 timeline（四大案名），t2 people（牵连人物，depends_on=[t1]），t3 timeline（后果，depends_on=[t1]）
- 不合格：t2 t3 都走 timeline — 没跨 people

**严格链式**：t1 → t2 → t3，沿链至少切换一次域。

- 合格：`"送给朱棣那顶白帽子的人是谁、此人投靠朱棣的关键转折发生在哪一年、这件事对靖难之役的发动有何意义？"`
  - t1 people（找人），t2 timeline（此人事件，depends_on=[t1]），t3 timeline 或 people（分析意义，depends_on=[t2]）
- 不合格：t1 t2 t3 都走 people — 没跨 timeline

**通用约束**：

- 每道题只允许一个连续问句，链式 / 扇出可用顿号连接子追问，但整体仍是一句。
- 每个 task 的 `intention` 必须能由其 `task` 文本独立决定（脱离上下游也能判定）。
- 题目不能让 LLM 直接靠常识回答，必须依赖工具检索 people.json + timeline.json 才能得出完整答案。
- 答案必须分别落在两个域的字段上，不得只用一个域就完整作答。

---

**⛔ 绝对禁止（以下规则无例外）**

**禁止一：单域题混入**

任何题目，若仅靠 people 或仅靠 timeline 就能完整作答，**不属于 orchestrator 范畴**，必须改写或丢弃。判别方式：删掉一个域的所有 keypoint，剩余 keypoint 能否完整回答 question？能 → 单域题，不合格。

**禁止二：问题里不得出现任何 JSON 字段名**

`people_id`、`event_id`、`primary_identity`、`secondary_identity`、`roles`、`relationships`、`era`、`year`、`outcome`、`participants`、`tags`、`summary`、`source_chunks` 等都是内部字段名，绝对不能出现在问题里。

**禁止三：问题里不得照搬字段原文**

`summary` / `relationships.context` / `outcome` 字段往往有完整表述，把这些句子直接嵌入问题会让向量检索直接命中原文，失去评测意义。

**禁止四：n 必须严格遵守**

每个子类型恰好出 `n` 道题，不得多出也不得少出。共 4 个子类型 × n 道。

---

### Step 3：写 expected_tasks + keypoints

每道题输出三个字段：`question` / `expected_tasks` / `keypoints`。

**`expected_tasks` 结构**：

```json
[
  {
    "task_id":     "t1",
    "task":        "子任务文本（用于人工审阅，自检不做严格匹配）",
    "query_kind":  "fact | multi_enum | analysis",
    "intention":   "people | timeline",
    "depends_on":  []
  }
]
```

`task_id` 从 `t1` 起连续编号，`depends_on` 引用必须出现在前面的 task_id。

**`keypoints` 结构**（每条新增 `task_id` 字段标识归属）：

```json
{
  "task_id":  "t1",
  "answer":   "能独立回答该 task 的最短完整表述",
  "source":   "people[X].FIELD 或 timeline[ID].FIELD 或 派生列表"
}
```

**关键约束**：

- 每条 keypoint 必须有 `task_id`，且必须在 `expected_tasks` 中存在
- 每个 `task` 至少要有 1 条 keypoint 支撑
- **source 域必须与 task intention 匹配**：`intention=people` → source 以 `people[` 开头；`intention=timeline` → source 以 `timeline[` 开头
- 派生 keypoint（如时间差、前后比较）source 可为列表，每个元素都必须匹配该 task 的 intention

**source 写法规则**：

| 情形 | source 格式 |
|---|---|
| 人物某字段 | `people[朱元璋].primary_identity` |
| 人物关系 | `people[朱元璋].relationships[徐达].context` |
| 事件年份 | `timeline[42].year` |
| 事件结果 | `timeline[42].outcome` |
| 派生（两个事件年份） | `["timeline[X].year", "timeline[Y].year"]` |

---

**正确示例（四种子类型各一）**

```json
// 并行跨域
{
  "qna_id":   "orch_永乐_001",
  "sub_type": "并行跨域",
  "question": "于谦参与了哪些重大历史事件、于谦与哪些人有决定性关联、书中最终如何评价于谦？",
  "expected_tasks": [
    {"task_id": "t1", "task": "于谦参与了哪些重大历史事件？",       "query_kind": "multi_enum", "intention": "timeline", "depends_on": []},
    {"task_id": "t2", "task": "于谦与哪些人有决定性关联？",         "query_kind": "multi_enum", "intention": "people",   "depends_on": []},
    {"task_id": "t3", "task": "书中最终如何评价于谦？",             "query_kind": "analysis",   "intention": "people",   "depends_on": []}
  ],
  "keypoints": [
    {"task_id": "t1", "answer": "北京保卫战",       "source": "timeline[XXX].event"},
    {"task_id": "t1", "answer": "土木堡之变善后",   "source": "timeline[XXX].event"},
    {"task_id": "t2", "answer": "石亨",             "source": "people[于谦].relationships[石亨].context"},
    {"task_id": "t3", "answer": "保全社稷的忠臣",   "source": "people[于谦].summary"}
  ]
}

// 顺序两级
{
  "qna_id":   "orch_永乐_002",
  "sub_type": "顺序两级",
  "question": "永乐年间最重要的几件大事分别发生在什么时候、各自涉及哪些核心人物？",
  "expected_tasks": [
    {"task_id": "t1", "task": "永乐年间最重要的几件大事分别发生在什么时候？", "query_kind": "multi_enum", "intention": "timeline", "depends_on": []},
    {"task_id": "t2", "task": "永乐年间最重要的几件大事各自涉及哪些核心人物？","query_kind": "multi_enum", "intention": "people",   "depends_on": ["t1"]}
  ],
  "keypoints": [
    {"task_id": "t1", "answer": "迁都北京", "source": "timeline[XXX].event"},
    {"task_id": "t1", "answer": "1421",     "source": "timeline[XXX].year"},
    {"task_id": "t2", "answer": "姚广孝",   "source": "people[姚广孝].roles"}
  ]
}

// 扇出枚举
{
  "qna_id":   "orch_永乐_003",
  "sub_type": "扇出枚举",
  "question": "永乐朝几次北征分别在什么时候发生、各由谁统帅、各自结果如何？",
  "expected_tasks": [
    {"task_id": "t1", "task": "永乐朝几次北征分别在什么时候发生？", "query_kind": "multi_enum", "intention": "timeline", "depends_on": []},
    {"task_id": "t2", "task": "永乐朝几次北征各由谁统帅？",         "query_kind": "multi_enum", "intention": "people",   "depends_on": ["t1"]},
    {"task_id": "t3", "task": "永乐朝几次北征各自结果如何？",       "query_kind": "multi_enum", "intention": "timeline", "depends_on": ["t1"]}
  ],
  "keypoints": [
    {"task_id": "t1", "answer": "朱棣第一次北征", "source": "timeline[XXX].event"},
    {"task_id": "t2", "answer": "丘福",           "source": "people[丘福].roles"},
    {"task_id": "t3", "answer": "本雅失里败走",   "source": "timeline[XXX].outcome"}
  ]
}

// 严格链式
{
  "qna_id":   "orch_永乐_004",
  "sub_type": "严格链式",
  "question": "送给朱棣那顶白帽子的人是谁、此人在朱棣登基过程中起到了什么关键作用、这种作用如何在永乐朝政治格局里延续？",
  "expected_tasks": [
    {"task_id": "t1", "task": "送给朱棣那顶白帽子的人是谁？",                  "query_kind": "fact",     "intention": "people",   "depends_on": []},
    {"task_id": "t2", "task": "此人在朱棣登基过程中起到了什么关键作用？",       "query_kind": "analysis", "intention": "timeline", "depends_on": ["t1"]},
    {"task_id": "t3", "task": "这种作用如何在永乐朝政治格局里延续？",           "query_kind": "analysis", "intention": "people",   "depends_on": ["t2"]}
  ],
  "keypoints": [
    {"task_id": "t1", "answer": "姚广孝",         "source": "people[姚广孝].aliases"},
    {"task_id": "t2", "answer": "靖难谋主",       "source": "timeline[XXX].participants"},
    {"task_id": "t3", "answer": "黑衣宰相",       "source": "people[姚广孝].roles"}
  ]
}
```

---

**错误示例**

```json
// ❌ 并行跨域：所有 task 都落在 timeline，没跨人物域
{
  "sub_type": "并行跨域",
  "question": "永乐年间发生了哪些战役、哪些远航、哪些政治大事？",
  "expected_tasks": [
    {"task_id": "t1", "intention": "timeline", "depends_on": []},
    {"task_id": "t2", "intention": "timeline", "depends_on": []},
    {"task_id": "t3", "intention": "timeline", "depends_on": []}
  ]
}
// 错误原因：intention 全是 timeline，违反跨域强制。
// 修法：把一条改成 people 域提问，如"涉及哪些核心将领"。

// ❌ 扇出枚举：扇出 task 没跨域
{
  "sub_type": "扇出枚举",
  "expected_tasks": [
    {"task_id": "t1", "intention": "timeline", "depends_on": []},
    {"task_id": "t2", "intention": "timeline", "depends_on": ["t1"]},
    {"task_id": "t3", "intention": "timeline", "depends_on": ["t1"]}
  ]
}
// 错误原因：t1/t2/t3 全 timeline，单域题。

// ❌ keypoint 域与 task intention 不匹配
{
  "expected_tasks": [
    {"task_id": "t1", "intention": "timeline", "depends_on": []}
  ],
  "keypoints": [
    {"task_id": "t1", "answer": "...", "source": "people[X].roles"}
  ]
}
// 错误原因：t1 intention=timeline 但 keypoint source 来自 people。
// 修法：要么改 source 到 timeline，要么把 task intention 改为 people。
```

---

### Step 4：写出 JSON

输出路径 = `dirname(people_json)/orchestrator_eval_{era_joined}.json`

其中 `era_joined` 是参数 era 的下划线连接，如 `"永乐"` → `orchestrator_eval_永乐.json`，`"建文,永乐"` → `orchestrator_eval_建文_永乐.json`。

`qna_id` 格式：`orch_{era_joined}_{nnn}`，从 `001` 起连续编号。

---

## Part 2：自检

JSON 写完后立即运行：

```bash
# 使用项目默认路径
python3 {SKILL_DIR}/self_check.py --eval_json {output_json_path}

# 使用自定义路径
python3 {SKILL_DIR}/self_check.py --eval_json {output_json_path} --people_json {people_json} --timeline_json {timeline_json}
```

脚本输出三部分，必须全部通过才算完成：

---

### Part 1：结构性硬约束

逐题打印六项结构性检查：

- **跨域强制**：`intention` 必须同时含 people 和 timeline，否则 `⚠️`
- **task_id 连续编号**：必须从 `t1` 起连续，不允许跳号（与 query_understanding 约定一致）
- **task_id 一致性**：每个 keypoint.task_id 必须在 expected_tasks 中，每个 task 至少 1 条 keypoint
- **depends_on 合法性**：引用的 task_id 必须存在且排在当前 task 之前
- **子类型拓扑合规**：四种 sub_type 各自的 depends_on 拓扑必须匹配
- **source-intention 匹配**：people task 的 source 必须 `people[` 开头，timeline 同理

任何 `⚠️` → 修改后重跑。

---

### Part 2：Keypoint 溯源验证

逐条解析 source 路径到具体字段值，并检测 answer 是否出现在字段值里。

- `✅` answer 在字段值中找到
- `⚠️` 未找到或 source 路径无效

出现 `⚠️` 的处理方式：

- answer 措辞与字段原文略有出入 → 修改 answer，贴近字段原文
- source 路径填错（人物名 / event_id / 字段名）→ 修正 source
- answer 是先验知识而非字段内容 → 删除该 keypoint 或换有字段支撑的事实

---

### Part 3：问题模板多样性检查

统计每个子类型内的疑问词分布。

- 同一子类型 `n` 道题全部使用同一疑问词 → `⚠️` 违规
- 全局任意一个疑问词占比 ≥ 70% → `⚠️` 违规

**可用备选问法**：

- 直接定位：`"送给朱棣白帽子的人是谁、此人后来发挥了什么作用？"`
- 时段枚举：`"永乐年间最关键的几桩政治变动是什么、各自由谁主导？"`
- 反向溯源：`"靖难之役的关键决策者是谁、他在哪一年加入朱棣阵营？"`
- 链式追问：`"洪都保卫战守住后，主将后来去了何处、为何最终被废？"`

---

自检全部通过后，终端打印：`✅ 三部分自检全部通过（共 {total} 道题）`
