# mingchao-kg-builder 执行逻辑

## 总体流程

```
用户提供: chunk JSON 路径 + chunk_id 范围 + 输出目录
                         │
                         ▼
             Step 0 · 确认辅助脚本
             ─────────────────────────────────
             检查 <SKILL_DIR>/scripts/ 下是否存在：
               script_chunk_extraction.py
               script_incremental_merge.py
               validate_kg.py
             ─────────────────────────────────
             ✅ 存在 → 继续
             ❌ 缺失 → 告知用户，不重新生成脚本
                         │
                         ▼
             Step 1 · 确定分批计划
             ─────────────────────────────────
             batch_size = 20
             1-91 → [1-20, 21-40, 41-60, 61-80, 81-91]
             告知用户分批方案
                         │
                         ▼
             ┌───────────────────────────────┐
             │  Step 2 · 逐批增量循环        │
             │  （对每一批重复 A → E）       │
             └───────────────────────────────┘
                         │
          ┌──────────────┘
          │
          │  A · 逐 chunk 提取原文
          │  ─────────────────────────────────
          │  python script_chunk_extraction.py <path> <id>
          │  × batch_size 次（每个 chunk_id 依次运行）
          │  在 LLM 上下文中累积分析：
          │    - 哪些人物满足收录门槛（≥2 chunk 有实质描述）
          │    - 哪些事件满足收录门槛（有名可命名/情节转折点）
          │
          │  B · 输出本批 patch
          │  ─────────────────────────────────
          │  Write 工具写出：
          │    patch_people.json    ← 本批新增/更新人物
          │    patch_timeline.json  ← 本批新增/更新事件
          │
          │  ⚠  summary 写法原则：
          │    首次出现的人物 → 写覆盖本批所有行动的 summary
          │    已有的人物（如朱元璋）→ 只写本批时段的新内容
          │    （合并脚本用 [PART] 拼接，Step 3-B 再统一合成）
          │  ⚠  summary 严禁使用代词（他/她/其/此人等）
          │    → 必须始终用规范名字显式指代（RAG 召回需要）
          │
          │  C · 增量合并写盘
          │  ─────────────────────────────────
          │  python script_incremental_merge.py \
          │    --people_patch    patch_people.json \
          │    --timeline_patch  patch_timeline.json \
          │    --output_people   chunk_{b_s}_{b_e}_people.json \
          │    --output_timeline chunk_{b_s}_{b_e}_timeline.json
          │
          │  脚本输出 3 行报告：
          │    people:   新增 X 人 / 更新 Y 人 → 当前共 Z 人
          │    timeline: 新增 X 件 / 更新 Y 件 → 当前共 Z 件
          │    写出完成: <路径>
          │
          │  D · 清理临时文件
          │  ─────────────────────────────────
          │  rm patch_people.json patch_timeline.json
          │
          │  E · Validate 自检（每批必须通过）
          │  ─────────────────────────────────
          │  python validate_kg.py \
          │    chunk_{b_s}_{b_e}_people.json \
          │    chunk_{b_s}_{b_e}_timeline.json
          │
          │    ✅ 全部通过 → 继续下一批
          │    ❌ 有报错  → 修复 → 重新 validate → 不能跳过
          │
          └──── 下一批 ────┐
                           ▼
                 所有批次通过后
                           │
                           ▼
             Step 3 · 最终合并与 summary 合成
             ─────────────────────────────────
```

---

## Step 3 详细流程

```
Step 3-A · 逐批合并成最终文件
─────────────────────────────────────────────
复制第一批文件为最终输出：
  chunk_{start}_{end}_people.json
  chunk_{start}_{end}_timeline.json

依次将第 2、3、... 批合并进去：
  python script_incremental_merge.py \
    --people_patch    chunk_{b_s}_{b_e}_people.json \
    --timeline_patch  chunk_{b_s}_{b_e}_timeline.json \
    --output_people   chunk_{start}_{end}_people.json \
    --output_timeline chunk_{start}_{end}_timeline.json
（重复至最后一批）
                    │
                    ▼
Step 3-B · Summary 合成（最重要）
─────────────────────────────────────────────
对最终文件中含 [PART] 分隔符的 summary，
LLM 根据各批片段内容改写为统一连贯摘要：
  - 覆盖所有批次的主要行动，不遗漏任何阶段
  - 按时间顺序组织
  - 主角 8-15 句 / 次要人物 3-5 句
  - 改写成连贯叙事，不拼凑原句
  - 严禁使用代词（他/她/其等），始终用规范名字
  - 最终文件不得出现任何 [PART]
                    │
                    ▼
Step 3-C · Roles 压缩
─────────────────────────────────────────────
roles 超过 5 项的人物 → 保留最核心 3-5 项
删去单次事件性的阶段角色（如"龙湾之战总指挥"）
                    │
                    ▼
Step 3-D · Relationships 去冗
─────────────────────────────────────────────
同一人物对同一 target 有多条 relationship：
  type 相同或相近 → 保留 context 最详尽的一条
  type 确实不同  → 均保留（但 context 不应重叠）
                    │
                    ▼
Step 3-E · 最终 Validate
─────────────────────────────────────────────
python validate_kg.py \
  chunk_{start}_{end}_people.json \
  chunk_{start}_{end}_timeline.json
✅ 通过 → 告知用户路径与人物/事件数量
❌ 报错 → 修复后重新运行
```

---

## Validate 自检内容

每批 validate 和最终 validate 均执行以下检查：

```
people.json 检查
  ├─ 必填字段齐全（9 个）
  │    aliases / primary_identity / secondary_identity /
  │    era / roles / relationships / events /
  │    source_chunks / summary
  ├─ primary_identity ∈ 13 值枚举（非空）
  │    皇帝 / 皇室 / 明朝武将 / 反叛势力 / 文臣 / 宦官 /
  │    社会人员 / 蒙古草原 / 势力 / 清 / 朝鲜 / 日本 / 外国
  ├─ secondary_identity 是列表
  ├─ era 不含描述性词（"明初"/"早年"），只写年号
  ├─ roles 非空
  ├─ relationships[i].context ≥ 12 字，无空洞表述
  └─ source_chunks 全为整数

timeline.json 检查
  ├─ 必填字段齐全（9 个）
  │    year / era / event / tags / location /
  │    participants / outcome / source_chunks / summary
  ├─ year 是整数
  ├─ era 长度 ≥ 4 字
  ├─ tags[0] ∈ 7 值类型枚举
  │    战役 / 起义 / 政治 / 外交 / 人物节点 / 民生 / 其他
  ├─ outcome ≥ 15 字，无空洞表述
  ├─ location 非空
  └─ source_chunks 全为整数

交叉引用检查
  ├─ people.events 中每个名称在 timeline.event 中存在
  ├─ timeline.participants 中每个名称在 people key 中存在
  └─ relationships.target 在 people key 中存在
```

---

## Summary 合成：朱元璋示例

### 增量循环后的 [PART] 拼接状态（待合成）

```
chunk_1_91：朱元璋幼名朱重八，出身钟离贫苦农家……
            1368年于应天称帝建明，最终由淮右贫民成长为
            新王朝的开创者与统一战争的主导者。
[PART]
chunk_92_170：进入洪武朝中后期后，朱元璋把国家建设、
              官僚整肃和皇权重塑同时推进……朱标的突然
              去世则彻底改变了朱元璋的晚年部署。
[PART]
chunk_171_263：朱元璋在本段中主要以洪武晚年的开国皇帝
               形象出现。洪武二十三年与二十九年两次命朱棣
               北征……洪武三十一年去世时，朱元璋相信诸王
               与新君能够共守祖制，却没意识到自己留下的
               恰恰是一套随时可能内爆的权力结构。
```

### ✅ Step 3-B 合成后的合格版本（零代词）

```
朱元璋，幼名朱重八，出身钟离贫苦农家。至正四年灾荒瘟疫并至，
父母兄侄相继饿死，朱元璋在无地葬亲的绝境中入皇觉寺为僧、流浪
讨饭，磨出极强的意志与观察力。至正十二年在汤和来信的促动下投
奔郭子兴，改名朱元璋，迎娶马秀英，并在濠州旧集团的倾轧中走向
独立。此后朱元璋连克定远、滁州，招得徐达等二十四人，得李善长
辅佐，至正十六年攻取集庆并改名应天，正式奠定争天下的核心基地。
龙湾之战以伏击重创陈友谅，后又历经鄱阳湖决战击灭陈汉、平江一
役消灭张士诚，北伐中原，于1368年正月称帝建明，年号洪武，徐达
北伐攻克大都、元顺帝出走。洪武年间，朱元璋在建设与整肃两条线
上同时发力：推行垦荒免税、恢复科举，确立八股取士制度；朝廷内
部则先借杨宪压制淮西、后纵容胡惟庸坐大，最终在洪武十三年借胡
惟庸案展开大清洗并废除丞相制度，把政务尽收于皇帝一身。此后空
印案、郭桓案、蓝玉案接连爆发，功臣宿将大量落网，锦衣卫与严刑
重典将皇权控制延伸至官僚体系每个角落。洪武晚年，朱元璋在北元
威胁与帝国继承两个方向上同步布局。洪武二十三年与二十九年两次
命朱棣北征，使朱棣的军事声望不断上升。朱标病逝后，朱元璋仍坚
持以嫡系孙支承继大统，立朱允炆为皇太孙。洪武三十一年朱元璋去
世，相信诸王与新君能共守祖制，却未意识到自己留下的恰好是一套
随时可能内爆的权力结构。
```

覆盖三批全部阶段，无 [PART]，无代词。

### ❌ 不合格示例（只取了 chunk_1_91 的内容）

```
朱元璋幼名朱重八，出身钟离贫苦农家……最终由淮右贫民成长为
新王朝的开创者与统一战争的主导者。
```

chunk_92_170 的洪武朝整肃（胡惟庸案、废丞相、四大案）和
chunk_171_263 的晚年布局（朱棣北征、蓝玉案、朱允炆储位）全部缺失。
字数最长的那批不等于覆盖最全，必须读完所有 [PART] 片段后重新改写。
