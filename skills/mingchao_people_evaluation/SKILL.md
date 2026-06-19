---
name: mingchao_people_evaluation
description: 输入人物库 JSON 路径、people_id 区间和每子类型题数 n，生成用于 People 检索压力测试的评测题库 JSON。覆盖身份筛选枚举、关系定位、官职→人物、并行枚举、链式推理五种子类型。
---

# People 检索题库生成器

## 参数

| 参数 | 是否必填 | 含义 | 默认 / 示例 |
|---|---|---|---|
| `people_json` | 可选 | people.json 路径，不填则使用项目标准路径 | 默认：`…/data/people_timeline/mingchao_people.json` |
| `people_id_start` | 必填 | 起始 people_id（含） |  |
| `people_id_end` | 必填 | 结束 people_id（含） |  |
| `n` | 可选 | 每子类型题数 | 默认 `2` |

---

## Part 1：生成题库

### Step 1：加载人物数据

严禁直接读取整个 people JSON（全量 185 条，一次性读会越界）。唯一方式：

```bash
# 使用项目默认路径
python3 {SKILL_DIR}/people_loader.py --start {people_id_start} --end {people_id_end}

# 使用自定义路径
python3 {SKILL_DIR}/people_loader.py --start {people_id_start} --end {people_id_end} --json_path {people_json}
```

把输出完整读进来，标记每位人物的 people_id、name 及所有字段内容。

---

### Step 2：按子类型出题

每个子类型出 `n` 道题。每道题必须能从**已加载的人物数据**里找到可溯源的锚点，严禁使用先验历史知识凭空填题。

**五种子类型及锚点策略**：

**身份筛选枚举**：扫 `primary_identity` / `secondary_identity` / `roles`，找同类标签下的多人集合，问"有哪些……"。答案是多个人名。

- 合格：`"明朝有哪些太监专权？"` — 扫 roles 含"专权"或 primary_identity 含"太监"的人物，枚举人名
- 不合格：`"明朝太监专权最早是哪一年？"` — 答案是年份，落在 Timeline 域，不属于 People

**关系定位**：扫 `relationships`，找 context 里有唯一性描述（"第一""最关键""唯一"）的关系条目，问"谁是……"。答案是一个人名。

- 合格：`"朱棣麾下最关键的谋士是谁？"` — relationships[道衍].context 有明确表述
- 不合格：`"朱元璋最信任谁？"` — 过于主观，无法从 relationships 唯一锚定

**官职→人物**：扫 `roles`，找含具体官职名且在加载区间内有唯一对应人物的条目，问"第一任……是谁"或"担任……的人是谁"。

- 合格：`"第一任征虏大将军是谁？"` — roles 含"征虏大将军"且仅此一人
- 不合格：`"谁担任过官职？"` — 太宽泛，无法指向唯一答案

**并行枚举（复杂）**：选一位人物，同时问其多个维度（动机 + 贡献 + 态度，或文臣 + 武将分类），每个维度对应独立的 keypoint 组。

- 合格：`"道衍辅佐朱棣的动机、造反中的具体贡献、以及朱棣对他的最终态度？"` — 三个维度各有字段支撑
- 不合格：把三个问题拆成三道独立题，然后合并 — 题目必须是一个连续追问，而非拼接

**链式推理（复杂）**：找 A→B 的 relationship，再从 B 的 roles 或 B 对 A 的 relationships 继续追问，形成"先找谁、此人再……"的连续问法。答案落点始终在人物域。

- 合格：`"送朱棣那顶白帽子的人是谁、此人在朱棣麾下担任了什么职务、这个职务对他日后权势有何作用？"` — 先定位道衍，再追问 roles 和 relationships
- 不合格：`"道衍是谁？"` — 没有链式，单跳直答

**通用约束**：
- 每道题只允许一个问句。链式推理例外，可用顿号连接追问，但整体是连续句，不得拆成两个独立问号结尾的句子。
- 答案落点必须在人物属性（名字、身份、职务、关系、数量）上，不得落在事件 / 时间 / 地点（那是 Timeline 的范畴）。例如"太监专权最早哪一年"的答案是年份，归 Timeline，不归 People；"朱元璋有多少儿子"的答案是人数，归 People。
- 题目不能让 LLM 直接靠常识回答，必须依赖工具检索 people.json 才能得出完整答案。

---

**⛔ 绝对禁止（以下规则无例外）**

**禁止一：问题里不得出现任何 JSON 字段名**

这是根据《明朝那些事儿》内容出的评测题，读者是真实用户，不是工程师。`primary_identity`、`secondary_identity`、`roles`、`relationships`、`aliases`、`people_id`、`summary`、`source_chunks` 等都是内部字段名，绝对不能出现在问题里。

| ❌ 不合格 | ✅ 合格 |
|---|---|
| `primary_identity 为反叛势力的人物有哪些？` | `元末有哪些著名的反叛势力首领？` |
| `secondary_identity 中带有红巾军身份的有谁？` | `哪些人曾以红巾军将领身份参战？` |
| `roles 中包含御史中丞的是谁？` | `谁担任过御史中丞？` |
| `people_id 1-30 中有哪些……` | `（直接问，不加任何 ID 范围限定）` |
| `在人物库中……` | `（也禁止，问题里不暴露数据库概念）` |

**禁止二：问题里不得照搬字段原文**

`relationships` / `summary` 字段往往有完整表述，把这些句子直接嵌入问题，会让向量检索直接命中原文，失去评测意义。

| ❌ 不合格 | ✅ 合格 |
|---|---|
| `被朱元璋长期委以主力军务、并以其为第一主将的是谁？` | `朱元璋最倚重的主力大将是谁？` |
| `暗中勾连张士诚、最终被废为庶人的宗室将领是谁？` | `守住洪都后却走上反叛之路的宗室将领是谁？` |

**禁止三：n 必须严格遵守**

每个子类型恰好出 `n` 道题，不得多出也不得少出。

---

### Step 3：写 keypoints

每道题写 `keypoints` 列表，每条 keypoint 两个字段：

- `answer`：**能独立回答该问题（或该跳）的最短完整表述**，必须能从 source 字段原文中读出。列举型（哪些人）写人名即可；动作型（谁做了什么、报给了谁）需包含主语＋动作＋对象，但不加任何多余背景。
- `source`：path 表示法，精确指向 people.json 里的具体字段。

**各子类型 answer 写法**：

| 子类型 | answer 写什么 | 条数上限 |
|---|---|---|
| 身份筛选枚举 | 每条写一个**人名**，source 指向该人含匹配标签的字段 | 每个符合的人一条 |
| 关系定位 | 第一条写**人名**，第二条可写支撑属性（可选） | 1-2 条 |
| 官职→人物 | 写**人名**；如需支撑可加一条属性 | 1-2 条 |
| 并行枚举 | 每个维度写该维度的具体事实（职务名、处置结果等） | 每维度 1 条，合计 ≥ 3 |
| 链式推理 | 每一跳写能独立成句的最短答案（含主语＋动作＋对象），一跳一条 | 每跳 1 条，合计 ≥ 2 |

**严禁写进 answer 的内容**：
- 来自 summary / relationships 的大段描述句（超出回答范围的冗余背景）
- 无法直接回答问题的条件从句（"洪都危急时"、"在某某时期"等）
- 用职务标签代替人名（问"哪些人"，answer 必须是人名，不能写"红巾军统帅"）
- 只写孤立关键词而不构成完整答案（问"派谁出城求援"，写"张子明"不够，需写"朱文正派张子明出城求援"）

**source 写法规则**示例：

| 情形 | source 格式 |
|---|---|
| 人物某字段 | `people[朱元璋].primary_identity` |
| roles 列表 | `people[徐达].roles` |
| aliases 列表 | `people[道衍].aliases` |
| summary 字段 | `people[张子明].summary` |
| 某条关系的 context | `people[朱元璋].relationships[徐达].context` |
| 派生值（两个来源） | `["people[A].roles", "people[B].relationships[A].context"]` |

> ⚠️ `people[X].relationships` 不是合法 source，必须精确到目标人物：`people[X].relationships[TARGET].context`

---

**正确示例（五种子类型各一）**

```json
// 身份筛选枚举：answer 是人名，source 证明该人有匹配标签
{
  "question": "元末反元阵营中，哪些人曾是红巾军体系的重要首领或统帅？",
  "keypoints": [
    { "answer": "朱元璋", "source": "people[朱元璋].roles" },
    { "answer": "韩山童", "source": "people[韩山童].roles" },
    { "answer": "刘福通", "source": "people[刘福通].roles" },
    { "answer": "徐寿辉", "source": "people[徐寿辉].roles" }
  ]
}

// 关系定位：只需一条人名，source 证明唯一性描述来自该字段
{
  "question": "陈友谅阵营里，哪位猛将最受他信赖？",
  "keypoints": [
    { "answer": "张定边", "source": "people[张定边].summary" }
  ]
}

// 官职→人物：答案只有一个人名，无需堆叠冗余支撑
{
  "question": "谁被称为明军第一名将，并担任北伐主帅？",
  "keypoints": [
    { "answer": "徐达", "source": "people[徐达].roles" }
  ]
}

// 并行枚举：每个维度一条，answer 是该维度的具体事实
{
  "question": "马秀英的出身关系、夫妻身份分别是什么？她在洪武朝保全了哪些人？",
  "keypoints": [
    { "answer": "出身为郭子兴义女",      "source": "people[马秀英].secondary_identity" },
    { "answer": "夫妻关系为朱元璋正室",      "source": "people[马秀英].roles" },
    { "answer": "保全李文忠、宋濂", "source": "people[马秀英].summary" }
  ]
}

// 链式推理：每一跳一条，answer 是该跳落点，不加无关背景
{
  "question": "朱文正派谁出城求援，这个人把洪都还能支撑的消息报给了哪位主帅？",
  "keypoints": [
    { "answer": "朱文正派张子明出城求援", "source": "people[朱文正].summary" },
    { "answer": "张子明把消息报告给了朱元璋", "source": "people[张子明].summary" }
  ]
}
```

---

**错误示例**

```json
// ❌ 身份筛选枚举：answer 写了职务标签，不是人名
{
  "question": "元末反元阵营中，哪些人曾是红巾军体系的重要首领或统帅？",
  "keypoints": [
    { "answer": "红巾军统帅", "source": "people[朱元璋].roles" },
    { "answer": "红巾军倡首", "source": "people[韩山童].roles" }
  ]
}
// 错误原因：问题问的是"哪些人"，落点是人名。
// answer 写"红巾军统帅"而非"朱元璋"，LLM 答出人名后无法与 keypoint 匹配，评测失效。

// ❌ 链式推理：answer 只写孤立关键词 + 混入无意义条件从句
{
  "question": "朱文正派谁出城求援，这个人把洪都还能支撑的消息报给了哪位主帅？",
  "keypoints": [
    { "answer": "张子明",    "source": "people[朱文正].summary" },
    { "answer": "朱元璋",    "source": "people[张子明].summary" },
    { "answer": "洪都危急时", "source": "people[张子明].summary" }
  ]
}
// 错误原因一："张子明"是孤立关键词，LLM 答出"朱文正派张子明出城求援"时无法匹配，
//   应写完整最短句"朱文正派张子明出城求援"。
// 错误原因二："洪都危急时"是条件从句，不是任何一跳的答案落点，纯属冗余。
```

---

### Step 4：写出 JSON

输出路径 = `dirname(people_json)/people_eval_{people_id_start}_{people_id_end}.json`

```json
[
  {
    "qna_id": "people_{people_id_start}-{people_id_end}_{三位数序号}",
    "sub_type": "people",
    "question": "…",
    "keypoints": [
      { "answer": "…", "source": "people[名字].字段名" }
    ]
  }
]
```

qna_id 从 001 起连续编号。

---

## Part 2：自检

JSON 写完后立即运行：

```bash
# 使用项目默认路径
python3 {SKILL_DIR}/self_check.py --eval_json {output_json_path}

# 使用自定义路径
python3 {SKILL_DIR}/self_check.py --eval_json {output_json_path} --people_json {people_json}
```

脚本输出两部分，必须全部通过才算完成：

---

### Keypoint 溯源验证

脚本逐题打印每个 keypoint 的 `answer` + `source` 指向的原始字段内容，并自动检测 answer 是否出现在字段值里。

- `✅` 表示 answer 在字段值中找到
- `⚠️` 表示 answer 未在字段值中找到，或 source 路径无效

出现 `⚠️` 的处理方式：
- answer 确实来自该字段但措辞略有不同 → 修改 answer，贴近字段原文
- source 路径填错 → 修正 source
- answer 是先验知识而非字段原文 → 删除该 keypoint 或换一个有字段支撑的事实

改完重新运行，直到全部 `✅`。

---

### 问题模板多样性检查

脚本统计每个子类型内的疑问词分布，并输出全局占比：

- 同一子类型的 n 道题全部使用同一疑问词 → `⚠️` 违规，必须改
- 全局任意一个疑问词占比 ≥ 70% → `⚠️` 违规，必须改

**可用备选问法**（结构要真正不同，不是换个疑问词加同一个状语）：
- 直接定位：`"第一任锦衣卫指挥使是谁？"`
- 条件枚举：`"洪武朝封公爵的开国功臣里有哪些人后来被清洗？"`
- 并列追问：`"汤和在明初的身份、立场、以及朱元璋对他的最终处置分别是什么？"`
- 反向溯源：`"是谁最先劝说朱元璋不要急着称帝？"`

出现 `⚠️` → 修改对应题目，重新运行，直到全部 `✅`。

---

自检全部通过后，终端打印：`已生成 {total} 道题，写入 {output_path}`
