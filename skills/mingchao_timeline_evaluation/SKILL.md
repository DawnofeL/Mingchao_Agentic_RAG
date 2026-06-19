---
name: mingchao_timeline_evaluation
description: 输入时间线 JSON 路径、event_id 区间和每子类型题数 n，生成用于 Timeline 检索压力测试的评测题库 JSON。覆盖时间段范围、精确查询、先后比较、持续时长、扇出枚举五种子类型。
---

# Timeline 检索题库生成器

## 参数

| 参数 | 是否必填 | 含义 | 默认 / 示例 |
|---|---|---|---|
| `timeline_json` | 可选 | timeline.json 路径，不填则使用项目标准路径 | 默认：`…/data/people_timeline/mingchao_timeline.json` |
| `event_id_start` | 必填 | 起始 event_id（含） |  |
| `event_id_end` | 必填 | 结束 event_id（含） |  |
| `n` | 可选 | 每子类型题数 | 默认 `2` |

---

## Part 1：生成题库

### Step 1：加载事件数据

严禁直接读取整个 timeline JSON（全量 258 条，一次性读会越界）。唯一方式：

```bash
# 使用项目默认路径
python3 {SKILL_DIR}/timeline_loader.py --start {event_id_start} --end {event_id_end}

# 使用自定义路径
python3 {SKILL_DIR}/timeline_loader.py --start {event_id_start} --end {event_id_end} --json_path {timeline_json}
```

把输出完整读进来，标记每条事件的 event_id、year、era、event 及所有字段内容。

---

### Step 2：按子类型出题

每个子类型出 `n` 道题。每道题必须能从**已加载的事件数据**里找到可溯源的锚点，严禁使用先验历史知识凭空填题。

**五种子类型及锚点策略**：

**时间段范围**：选一个 `era` 或年份区间，枚举该范围内的重要事件名，问"X 年间有哪些重大事件"或"X 朝发生了几次……"。答案是一组事件名。

- 合格：`"永乐年间有哪些重大事件？"` — era=永乐 的所有事件枚举
- 不合格：`"明朝有哪些大事？"` — 范围太宽，无法从区间数据完整覆盖

**精确查询**：给定一个精确锚点，找出对应的单一事实。两种方向：

- 时间→事件：选 `year` 精确（该年仅 1-2 条事件）的条目，问"X 年发生了什么大事"，答案是 event 名。
- 序数→年份/地点：找 `event` 字段含序数词（"第X次""首次"）的条目，问"第X次……是哪一年/在哪里"，答案是 year 或 location。

严禁问"如何/为何"——过程性答案在 chunk 里，timeline 字段没有这种信息。

- 合格：`"1344年，朱元璋人生中发生了什么关键变故？"` — year=1344 精确，取 event 字段
- 合格：`"傅友德七战七捷发生于哪一年？"` — event 含序数描述，取 year 字段
- 不合格：`"北元第一次大反扑是如何被挫败的？"` — 答案需要过程叙述，归 Chunk 域

**先后比较**：找同类型的两个事件，各取 `year` 比较，问"X 和 Y 谁先……"。keypoint text 写出推导结论，source 用列表标注两个 year 来源。

- 合格：`"陈友谅和张士诚谁先被消灭？"` — 取两人覆灭事件的 year 做比较
- 不合格：两个事件年份相差不超过 1 年，或者两个时间没有明确提到日期 — 结论不够明确，不适合出题

**持续时长**：找同一系列的开始事件和结束事件，各取 `year`，用差值回答"打了多少年""历时多久"。keypoint text 写推导结果（如"约四年（1399-1402）"），source 用列表。

- 合格：`"靖难之役持续了多少年？"` — 靖难开战（1399）和结束（1402）各一条，差值 = 3 年
- 不合格：开始和结束事件不在加载区间内 — 不得用先验知识补全，必须都在区间里

**扇出枚举（复杂）**：选若干相关事件（同年、同战役、同人物视角），对每个事件并行追问 year / location / outcome 中的多个维度，问"这几件事分别是什么、各在哪里发生、各自结果如何"。每个事件每个维度各出一条 keypoint。

- 合格：`"1360年朱元璋集团与陈友谅局势急转时，相关大事各在何处发生？"` — 多个事件各取 event + location
- 不合格：把多个事件的所有维度信息混入一条 keypoint，无法原子化溯源

**通用约束**：
- 每道题只允许一个问句，禁止"……，又……？"拼接。扇出枚举例外，可用顿号连接追问子属性，但整体是一个连续句。
- 答案必须落在 `year` / `event` / `location` / `outcome` / `participants` 字段上。凡是答案需要解释过程、方法、原因的（"如何被挫败""为什么失败"），归 Chunk 域，timeline 不出这类题。
- 答案落点不得是人物身份或人物列举（那是 People 的范畴）。例如"永乐年间最出名的太监是谁"归 People；"永乐年间有哪些重大事件"归 Timeline。
- 题目不能让 LLM 直接靠常识回答，必须依赖工具检索 timeline.json 才能得出完整答案。

---

**⛔ 绝对禁止（以下规则无例外）**

**禁止一：问题里不得出现任何 JSON 字段名**

这是根据《明朝那些事儿》内容出的评测题，读者是真实用户，不是工程师。`event_id`、`era`（作为字段名出现）、`event`（作为字段名出现）、`location`、`participants`、`outcome`、`summary`、`source_chunks`、`tags` 等都是内部字段名，绝对不能出现在问题里。

| ❌ 不合格 | ✅ 合格 |
|---|---|
| `era 为洪武的事件有哪些？` | `洪武年间发生了哪些重大事件？` |
| `participants 包含朱元璋的事件有哪些？` | `朱元璋亲自参与了哪些重要战役？` |
| `outcome 字段中提到胜利的事件有哪些？` | `哪些战役以明军大胜告终？` |
| `event_id 1-30 中有哪些……` | `（直接问，不加任何 ID 范围限定）` |

**禁止二：问题里不得照搬字段原文**

`outcome` / `summary` 字段往往有完整表述，把这些句子直接嵌入问题，会让向量检索直接命中原文，失去评测意义。

| ❌ 不合格 | ✅ 合格 |
|---|---|
| `"以少胜多击败陈友谅水军"发生在哪一年？` | `鄱阳湖之战是哪一年？` |
| `"宣告元朝在中原统治正式结束"的事件是什么？` | `元朝在中原的统治是哪一年正式终结的？` |

**禁止三：n 必须严格遵守**

每个子类型恰好出 `n` 道题，不得多出也不得少出。

---

### Step 3：写 keypoints

每道题写 `keypoints` 列表，每条 keypoint 两个字段：

- `answer`：**能独立回答该问题（或该枚举项）的最短完整表述**，必须能从 source 字段原文中读出。枚举型（哪些事件）写事件名即可；比较/时长型写含两端时间的完整结论句；扇出型写维度值。
- `source`：path 表示法，精确指向 timeline.json 里的具体字段。

**各子类型 answer 写法**：

| 子类型 | answer 写什么 | 条数 |
|---|---|---|
| 时间段范围 | 每条写一个事件名，覆盖该时间段内主要事件 | ≥ 3 条 |
| 精确查询 | 写含锚点和结果的完整句（"X年发生了Y"或"第X次...是Y年"） | 1 条 |
| 先后比较 | 写含两事件名和年份的完整比较句 | 1 条（派生，source 为列表） |
| 持续时长 | 写含起止年份和时长差值的完整句 | 1 条（派生，source 为列表） |
| 扇出枚举 | 每个事件每个维度一条；维度值即答案（事件名、地点名、结果等） | 每事件×维度数 |

**严禁写进 answer 的内容**：
- 需要过程叙述的内容（"如何""为何"类答案只能来自 chunk，不在 timeline 范畴）
- 孤立数字或片段（不能只写"1402"，需写"靖难之役结束于1402年"）
- 与问题无关的背景补充

**source 写法规则**：

| 情形 | source 格式 |
|---|---|
| 事件年份 | `timeline[42].year` |
| 事件名称 | `timeline[42].event` |
| 事件地点 | `timeline[42].location` |
| 事件结果 | `timeline[42].outcome` |
| 事件参与者 | `timeline[42].participants` |
| 派生值（差值、比较，两个来源） | `["timeline[X].year", "timeline[Y].year"]` |

---

**正确示例（五种子类型各一）**

```json
// 时间段范围：每条 answer 是一个事件名
{
  "question": "至正年间，元末反元阵营中发生了哪些关键起义或建政事件？",
  "keypoints": [
    { "answer": "韩山童刘福通起义",   "source": "timeline[3].event" },
    { "answer": "徐寿辉建立天完政权", "source": "timeline[4].event" },
    { "answer": "朱元璋投奔郭子兴",   "source": "timeline[5].event" }
  ]
}

// 精确查询：给定年份锚点，answer 是事件名（完整且足够独立）
{
  "question": "1344年，朱元璋人生中发生了什么关键变故？",
  "keypoints": [
    { "answer": "朱元璋家破入皇觉寺", "source": "timeline[2].event" }
  ]
}

// 先后比较：answer 是含两端年份的完整比较结论句
{
  "question": "高邮之战和龙湾之战哪一场先发生？",
  "keypoints": [
    { "answer": "高邮之战（1353年）早于龙湾之战（1360年）",
      "source": ["timeline[9].year", "timeline[15].year"] }
  ]
}

// 持续时长：answer 是含起止年份和时长的完整句
{
  "question": "朱元璋从投奔郭子兴到攻克应天，中间相隔多少年？",
  "keypoints": [
    { "answer": "从1352年投奔郭子兴到1356年攻克应天，相隔4年",
      "source": ["timeline[5].year", "timeline[11].year"] }
  ]
}

// 扇出枚举：多事件多维度，每个维度一条，answer 是该维度的值
{
  "question": "1360年朱元璋集团与陈友谅局势急转时，相关大事各自是什么、发生在何处？",
  "keypoints": [
    { "answer": "陈友谅弑徐寿辉建汉", "source": "timeline[13].event" },
    { "answer": "采石五通庙",          "source": "timeline[13].location" },
    { "answer": "刘基归附朱元璋",      "source": "timeline[14].event" },
    { "answer": "处州",                "source": "timeline[14].location" },
    { "answer": "龙湾之战",            "source": "timeline[15].event" },
    { "answer": "龙湾",               "source": "timeline[15].location" }
  ]
}
```

---

**错误示例**

```json
// ❌ 精确查询：问"如何"，答案落在 chunk 域而非 timeline 字段
{
  "question": "北元在大都失守后的第一次大反扑，是如何被挫败的？",
  "keypoints": [
    { "answer": "太原夜袭战", "source": "timeline[28].event" }
  ]
}
// 错误原因："如何被挫败"需要过程叙述，timeline 字段只有事件名和年份，
// 没有战术细节，真正的答案在 chunk 文本里。
// 精确查询只能问"哪一年""在哪里""发生了什么"，不能问"如何""为何"。

// ❌ 精确查询：answer 只写孤立片段，不构成完整答案
{
  "question": "1344年，朱元璋人生中发生了什么关键变故？",
  "keypoints": [
    { "answer": "皇觉寺", "source": "timeline[2].event" }
  ]
}
// 错误原因："皇觉寺"是地名片段，无法独立回答"发生了什么"。
// 应写完整事件名："朱元璋家破入皇觉寺"。
```

---

### Step 4：写出 JSON

输出路径 = `dirname(timeline_json)/timeline_eval_{event_id_start}_{event_id_end}.json`

```json
[
  {
    "qna_id": "timeline_{event_id_start}-{event_id_end}_{三位数序号}",
    "sub_type": "timeline",
    "question": "…",
    "keypoints": [
      { "answer": "…", "source": "timeline[N].字段名" }
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
python3 {SKILL_DIR}/self_check.py --eval_json {output_json_path} --timeline_json {timeline_json}
```

脚本输出两部分，必须全部通过才算完成：

---

### Keypoint 溯源验证

脚本逐题打印每个 keypoint 的 `answer` + `source` 指向的原始字段内容，并自动检测 answer 是否出现在字段值里。派生值（持续时长、先后比较）只核对 source 路径是否有效，answer 合理性由人工判断。

- `✅` 表示 answer 在字段值中找到，或 source 路径有效（派生值）
- `⚠️` 表示 answer 未在字段值中找到，或 source 路径无效

出现 `⚠️` 的处理方式：
- answer 措辞与字段原文略有出入 → 修改 answer，贴近字段原文
- source 路径填错（event_id 不存在、字段名拼错）→ 修正 source
- answer 是先验知识而非字段内容 → 删除该 keypoint 或换有字段支撑的事实

改完重新运行，直到全部 `✅`。

---

### 问题模板多样性检查

脚本统计每个子类型内的疑问词分布，并输出全局占比：

- 同一子类型的 n 道题全部使用同一疑问词 → `⚠️` 违规，必须改
- 全局任意一个疑问词占比 ≥ 70% → `⚠️` 违规，必须改

**可用备选问法**（结构要真正不同）：
- 数字锚定：`"郑和第三次下西洋是哪一年？"`
- 时间窗口：`"永乐年间发生了哪些影响深远的大事？"`
- 先后对比：`"靖难之役和朱棣迁都北京哪个先发生？"`
- 时长推算：`"郑和下西洋前后历时多少年？"`
- 扇出追问：`"洪武四大案分别是什么时候发生的、各自造成了什么后果？"`

出现 `⚠️` → 修改对应题目，重新运行，直到全部 `✅`。

---

自检全部通过后，终端打印：`已生成 {total} 道题，写入 {output_path}`
