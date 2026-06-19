---
name: mingchao_llm_assessment
description: 输入一个明朝 RAG TestResults JSON 文件，对每道题的 llm_results 按 keypoints 进行 0/1 二元评分，且一次任务只能输出一个最终 Eval JSON。用于用户要求”执行批改””批改 TestResults””评测 llm_results””生成 Eval”等场景；严格禁止生成 slim、中间文件、日志文件或任何非最终输出文件。
---

# Mingchao LLM Assessment Skill

你是 judge 模型。对每道题，根据 `keypoints` 判断 `llm_results` 是否命中，输出每条 keypoint 的 0/1 评分及聚合 `score` 字典。一次批改任务只能产生一个最终 `_Eval_*.json` 文件。

## 硬性约束

- 只接受一个输入 JSON 文件。
- 只允许输出一个最终 JSON 文件。
- 禁止创建或保留任何中间文件，包括但不限于 slim、intermediate、cache、log、统计文件。
- 禁止调用 `extractor.py`，该脚本已废弃。
- 禁止输出 `*_TestResults.json` 这类精简文件。
- 原子写入时可以短暂创建同名 `.tmp` 文件，但必须在 `os.replace` 后消失；异常时必须尽量清理 `.tmp`。
- `self_check.py` 只能读取最终 Eval 文件并打印终端结果，不允许写任何文件。

## 输入输出命名

输入文件名必须包含 `_TestResults_`：

```text
{prefix}_TestResults_{mode}.json
```

最终输出文件名必须只把第一次出现的 `_TestResults_` 替换为 `_Eval_`：

```text
{prefix}_Eval_{mode}.json
```

示例：

```text
timeline_eval_1_258_TestResults_agentic.json
-> timeline_eval_1_258_Eval_agentic.json

timeline_eval_1_258_TestResults_vector.json
-> timeline_eval_1_258_Eval_vector.json
```

命名逻辑必须等价于：

```python
name = input_path.name
if "_TestResults_" not in name:
    raise ValueError("输入文件名必须包含 _TestResults_")
output_name = name.replace("_TestResults_", "_Eval_", 1)
```

## 执行流程

1. 读取用户传入的原始 TestResults JSON。
2. 在内存中抽取评分所需字段，不落盘：
   - `qna_id`
   - `sub_type`（如果原始数据存在则保留）
   - `question`
   - `keypoints`，至少保留每个 keypoint 的 `answer`
   - `llm_results`
3. 按顺序逐题评分：对每条 keypoint 填写 `score` 字段（0 或 1），最后汇总聚合 `score` 字典；完成后立即追加到内存结果列表，并原子写入唯一最终 Eval 文件。
4. 全部完成后运行：

```bash
python self_check.py <唯一最终 Eval 文件>
```

5. 确认本次任务没有产生除唯一最终 Eval 文件外的任何输出文件；如发现 `.tmp` 残留，清理后再结束。

## 输出条目格式

每道题输出为原始题目的精简评分条目：

```json
{
  “qna_id”: “1”,
  “sub_type”: “timeline”,
  “question”: “问题文本”,
  “keypoints”: [
    {“answer”: “最短必答内容”, “score”: 1},
    {“answer”: “另一项内容”,   “score”: 0}
  ],
  “llm_results”: “被测系统回答”,
  “score”: {“0”: 1, “1”: 1}
}
```

- `keypoints[i].score`：该条 keypoint 的命中结果，`1` 表示命中，`0` 表示漏答。
- `score`：聚合计数，必须与 keypoints 完全一致：
  - `score[“0”]` = keypoints 中 score 为 0 的条目数
  - `score[“1”]` = keypoints 中 score 为 1 的条目数

如果原始条目没有 `sub_type`，可以省略该字段。

## 评分规则

### 0 / 1：keypoint 粒度判定

对每条 keypoint，判断 `llm_results` 中能否找到该 `keypoint.answer` 的等价表达，将结果写入该条目的 `score` 字段。

记 `1`（命中）：

- 字面一致。
- 同义改写。
- 合理换序。
- 同一人物、地点、事件的别名。
- 年号、公历年、历史语境中明确等价的时间表达。

记 `0`（漏答）：

- 完全未提。
- 答错。
- 只回答上位概念，缺少 keypoint 要求的具体实体、地点、事件、年份或结果。
- 明确说”无法回答””不知道”，且没有给出 keypoint 的等价表达。

计数约束：

```text
score[“0”] + score[“1”] == len(keypoints)
```

## 空值与异常

- `llm_results` 为空字符串或 null：所有 keypoint 记 `0`。
- `keypoints` 为空列表：停止处理并报错；不要为该题编造评分。
- 输入文件名不含 `_TestResults_`：停止处理并报错。
- 批改中断后再次执行时，可以覆盖同名最终 Eval 文件，但仍不得产生其他输出文件。
