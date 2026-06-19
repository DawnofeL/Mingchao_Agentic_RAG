# mingchao-people-timeline-complier 执行逻辑

## 用途定位

builder 处理"单卷分批 chunk 合并"，merger 处理"片段级 patch 合并"，complier 处理**卷级合并**——把单卷 merger 跑完的产物合并进多卷累计大 JSON。

适用场景：第三卷 people/timeline 已经做完，要把它合并进前两卷的累计大 JSON，得到前三卷的累计大 JSON。后续每卷做完都按这套流程往累计 JSON 里加。

---

## 设计核心

**脚本做路由，LLM 做阅读理解。**

- 脚本只负责：遍历新条目 → 在旧 JSON 中找候选 → 分流（直接写入 / 交给 LLM）
- LLM 负责：阅读新条目 + 候选老条目 → 判断是否同一对象 → 按守则做字段并集 → 直接写盘

没有 `__CONFLICT__` 标记，没有 `[PART]` 占位，没有中间格式转写。LLM 面对完整原始条目，结果直接落到 staged 文件。遇到无法自行决断的冲突，停下问用户。

---

## 总体流程

```
用户提供: 旧大 JSON 路径 + 新卷 JSON 路径 + 输出目录
                       │
                       ▼
           Step 0 · 路径确认
           ─────────────────────────────────
           确认四份 JSON 存在且字段齐全
           源文件只读，不做任何写操作
                       │
                       ▼
           Step 1 · 脚本候选筛选
           ─────────────────────────────────
           python script_find_candidates.py ...

           people 候选检测（5 类）：
             exact_key / key_in_aliases / aliases_overlap /
             aliases_intersection / name_similarity（兜底）

           timeline 候选检测（2 类）：
             same_year_name_close /
             same_year_participants_overlap

           机械处理：
             老条目 → staged 文件（原封不动复制）
             新条目无候选 → staged 文件（直接 append）

           输出 merge_workload.md：
             含所有"需 LLM 处理"条目及其完整候选
                       │
                       ▼
           ┌─────────────────────────────────┐
           │    Step 2 · LLM 处理            │
           │    merge_workload.md            │
           │                                 │
           │ 逐条：                          │
           │  阅读新条目 + 候选老条目         │
           │  判断同一性                     │
           │    是 → 字段并集 → Edit staged  │
           │    否 → append 进 staged        │
           │    模棱两可 → 停问用户           │
           │                                 │
           │ 遇停问情形，一次性攒齐再问       │
           └─────────────────────────────────┘
                       │
                       ▼
           Step 3 · validate_kg.py
           ─────────────────────────────────
           python validate_kg.py ...
           检查：字段齐全、枚举合法、引用闭环
           ✅ 全部通过 → 继续
           ❌ 有报错  → LLM 修复后重跑
                       │
                       ▼
           Step 4 · 改名、清理与汇报
           ─────────────────────────────────
           staged 改名为正式文件名
           rm merge_workload.md
           输出目录有且仅有两个新文件

           输出合并汇报：
             人物统计（旧 X / 新增 +P / 合并 -M / 最终 Y）
             事件统计（旧 X / 新增 +Q / 合并 -M / 最终 Y）
             人物重叠详情（每对：旧 key + 新 key → 统一 key + source_chunks 范围）
             事件重叠详情（每对：旧 event + 新 event → 统一 event + year + source_chunks 范围）
```

---

## 停问触发的 6 种情形

LLM 必须停下来问用户的场景：

```
1. primary_identity 跨大类冲突
   （武将↔宦官、文臣↔反叛势力、宦官↔皇室、外国↔本国等）

2. tags[0] 冲突
   （7 枚举之间，罕见，一律问）

3. timeline year 差 >1 年但提示同一事件

4. timeline location 真正矛盾（非别名扩写）

5. 同一性弱证据模棱两可
   （era 不重叠、primary 不一致、events 无重叠）

6. relationships 中同 target 含义对立
   （"辅佐 vs 反叛"、"任命 vs 处决"等数据矛盾）
```

---

## 关键设计原则

**dedupe 保留，cap 全部去掉**

跨卷 RAG 召回需要密度，不设字数和数量上限：

- `roles`：union dedupe，不限项数，合并同时期同职能的近义写法
- `relationships`：按 (type, target) 去重，不限条数，type 语义近义合并，性质不同保留
- `summary`：覆盖所有卷的全部行动，不限句数
- `aliases`：全部保留，越多越好

**零代词原则**

所有文本字段（summary、outcome、context）禁止使用代词，始终用规范名字。这是 RAG 召回精度的基础保障。

**源文件只读**

四份输入 JSON 全程禁止修改、覆盖、删除、重命名。唯一的输出是最终合并 JSON 两份，不创建任何 backup，不留任何其他产物。

---

## 中间文件清单

| 文件 | 生成步骤 | 用途 | 清理时机 |
|---|---|---|---|
| `*.staged.json` | Step 1 | 合并中间产物 | Step 4 改名为正式文件 |
| `merge_workload.md` | Step 1 | 需 LLM 处理的候选清单 | Step 4 删除 |

---

## 与 builder / merger 的关系

- **builder**：从单卷 chunk 原文出发，分批提取 → 增量合并 → summary 合成，产出**单卷 KG**
- **merger**：把若干分段 KG 文件合并成一份单卷 KG
- **complier**：把单卷 KG 合并进多卷累计 KG，产出**新的多卷累计 KG**

三个 skill 各管一段，合作覆盖从原文到多卷累计大 JSON 的全链路。
