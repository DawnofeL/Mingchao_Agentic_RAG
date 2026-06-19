#!/usr/bin/env python3
"""题库自检：GT 配对验证 + 句型连续检查。

用法：
    python3 self_check.py --eval_json <生成的题库JSON> --chunk_json <原始chunkJSON>
"""
import json
import re
import argparse

# 长词优先，避免"为什么"被"为何"截断
_Q_PATTERNS = [
    ('为什么', r'为什么'),
    ('为何',   r'为何'),
    ('怎样',   r'怎样'),
    ('怎么',   r'怎么'),
    ('如何',   r'如何'),
    ('哪里',   r'哪里'),
    ('哪些',   r'哪些'),
    ('哪',     r'哪.'),
    ('什么',   r'什么'),
    ('谁',     r'谁'),
]


def detect_q_word(question: str) -> str:
    for label, pattern in _Q_PATTERNS:
        if re.search(pattern, question):
            return label
    return '其他'


def load_json(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def part1_gt_pairing(entries: list, chunks: dict):
    print('=' * 68)
    print('Part 1  GT 配对验证  —  逐题确认 chunk 原文能直接回答问题')
    print('=' * 68)
    keypoint_errors = 0
    for e in entries:
        print(f"\n[{e['qna_id']}]  {e['sub_type']}")
        print(f"问：{e['question']}")
        for cid in e['gt_chunk_ids']:
            chunk = chunks.get(cid)
            if chunk is None:
                print(f"  ⚠️  GT chunk_id={cid} 在 chunk_json 中不存在")
            else:
                preview = chunk['content'][:300].replace('\n', ' ')
                print(f"  GT[{cid}]：{preview}…")

        # 单GT硬校验：gt_chunk_ids 必须恰好 1 个
        gt_ids = e.get('gt_chunk_ids', [])
        if len(gt_ids) != 1:
            print(f"  ⚠️  gt_chunk_ids 长度为 {len(gt_ids)}，必须恰好 1 个（只允许单 GT）")
            keypoint_errors += 1

        # keypoint 校验
        kps = e.get('keypoints')
        if not kps:
            print(f"  ⚠️  缺少 keypoints 字段或为空列表")
            keypoint_errors += 1
        else:
            for i, kp in enumerate(kps):
                answer = kp.get('answer', '').strip()
                if not answer:
                    print(f"  ⚠️  keypoint[{i}] answer 为空")
                    keypoint_errors += 1
                else:
                    print(f"  kp[{i}]：{answer}")

        print('  ' + '─' * 64)

    print()
    if keypoint_errors == 0:
        print('✅ keypoint 校验通过')
    else:
        print(f'⚠️  共发现 {keypoint_errors} 处 keypoint 问题，必须修改后重新自检')


def part2_sentence_pattern(entries: list):
    print('\n' + '=' * 68)
    print('Part 2  句型连续检查  —  连续 3 题同句型为违规，自动标注')
    print('=' * 68)

    total_violations = 0
    for sub_type in ('description', 'causation'):
        group = [e for e in entries if e['sub_type'] == sub_type]
        if not group:
            continue

        labels = [(e['qna_id'], detect_q_word(e['question'])) for e in group]

        print(f'\n{sub_type}：')
        for qna_id, word in labels:
            print(f'  {qna_id}  {word}')

        violations = [
            (labels[i-2][0], labels[i-1][0], labels[i][0], labels[i][1])
            for i in range(2, len(labels))
            if labels[i][1] == labels[i-1][1] == labels[i-2][1]
        ]

        if violations:
            total_violations += len(violations)
            for v in violations:
                print(f'  ⚠️  连续3题「{v[3]}」：{v[0]} / {v[1]} / {v[2]}')
        else:
            print('  ✅ 无连续3题同句型')

    print()
    if total_violations == 0:
        print('✅ 句型检查通过')
    else:
        print(f'⚠️  共发现 {total_violations} 处句型违规，必须修改后重新自检')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_json',  required=True, help='生成的题库 JSON 路径')
    parser.add_argument('--chunk_json', required=True, help='原始 chunk JSON 路径')
    args = parser.parse_args()

    entries = load_json(args.eval_json)
    chunks  = {item['chunk_id']: item for item in load_json(args.chunk_json)}

    part1_gt_pairing(entries, chunks)
    part2_sentence_pattern(entries)


if __name__ == '__main__':
    main()
