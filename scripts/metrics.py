#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本 AI 味体检 —— 代理指标计算

用法:
    python metrics.py 文本.txt
    python metrics.py 文本.txt --json
    python metrics.py --compare 原文.txt 改写后.txt
    echo "文本" | python metrics.py -

说明:
    这些是**代理指标**，不是检测器。它们衡量的是"人味"的统计特征
    （句长节奏、模板词、标点多样性等），与朱雀/GPTZero 的实际打分
    相关但不等价。参考区间是经验值，建议用真实检测结果自行校准。

零第三方依赖。
"""

import argparse
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------
SENT_END = "。！？!?…"
SHORT_SENT = 15          # 短句阈值（非空白字符）
LONG_SENT = 40           # 长句阈值
SMALL_SAMPLE_SENTS = 20  # 句数低于此值时，比例类指标噪声大
SMALL_SAMPLE_CHARS = 500

# 模板词 / 套话（AI 高频，人类低频）
TEMPLATE_PATTERNS = [
    (r"首先", "首先"),
    (r"其次", "其次"),
    (r"再次", "再次"),
    (r"最后[，,]", "最后，"),
    (r"综上所述", "综上所述"),
    (r"总而言之", "总而言之"),
    (r"总的来说", "总的来说"),
    (r"总体而言", "总体而言"),
    (r"值得注意的是", "值得注意的是"),
    (r"需要指出的是", "需要指出的是"),
    (r"需要注意的是", "需要注意的是"),
    (r"不难看出", "不难看出"),
    (r"显而易见", "显而易见"),
    (r"众所周知", "众所周知"),
    (r"毋庸置疑", "毋庸置疑"),
    (r"不可否认", "不可否认"),
    (r"毫无疑问", "毫无疑问"),
    (r"由此可见", "由此可见"),
    (r"与此同时", "与此同时"),
    (r"从某种意义上(说|讲)", "从某种意义上说"),
    (r"在一定程度上", "在一定程度上"),
    (r"在当今社会", "在当今社会"),
    (r"随着[^，。]{2,12}的(?:快速)?发展", "随着…的发展"),
    (r"在[^，。]{2,12}的背景下", "在…的背景下"),
    (r"不仅[^，。]{2,30}而且", "不仅…而且"),
    (r"一方面[^，。]{2,40}另一方面", "一方面…另一方面"),
    (r"发挥(?:着)?(?:重要|关键)作用", "发挥重要作用"),
    (r"起到(?:了)?(?:重要|关键)作用", "起到重要作用"),
    (r"具有(?:重要|深远)意义", "具有重要意义"),
    (r"提供(?:了)?有力支撑", "提供了有力支撑"),
    (r"奠定(?:了)?坚实基础", "奠定了坚实基础"),
    (r"注入(?:了)?新(?:的)?动能", "注入了新动能"),
    (r"迈上(?:了)?新台阶", "迈上新台阶"),
    (r"不可或缺", "不可或缺"),
    (r"日益(?:广泛|普遍|重要)", "日益…"),
    (r"高度(?:重视|关注)", "高度重视"),
    (r"深入推进", "深入推进"),
    (r"持续(?:优化|完善|提升)", "持续优化"),
    (r"有效(?:提升|促进|推动)", "有效提升"),
]

# 标点种类（用于多样性统计）
PUNCT_KINDS = "，。、；：？！…—～·《》〈〉「」『』（）【】〔〕“”‘’"

# 常见虚词（用于词频观察，不做"好坏"判断）
FUNCTION_WORDS = ["的", "了", "是", "在", "和", "有", "就", "不", "也", "与", "这", "那", "而", "被", "把", "对"]


# ---------------------------------------------------------------------------
# 基础切分
# ---------------------------------------------------------------------------
def split_sentences(text):
    """按句末标点和换行切句。"""
    out, buf = [], ""
    for ch in text:
        if ch == "\n":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
            continue
        buf += ch
        if ch in SENT_END:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def nchars(s):
    """非空白字符数。"""
    return len(re.sub(r"\s", "", s))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cv(xs):
    """变异系数 = 标准差 / 均值。衡量"起伏"。"""
    m = mean(xs)
    return stdev(xs) / m if m else 0.0


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------
def analyze(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sents = split_sentences(text)
    paras = split_paragraphs(text)
    total = nchars(text)

    slens = [nchars(s) for s in sents if nchars(s) > 0]
    plens = [nchars(p) for p in paras]

    # --- 句长节奏 ---
    rhythm = {
        "count": len(slens),
        "mean": round(mean(slens), 1),
        "stdev": round(stdev(slens), 1),
        "cv": round(cv(slens), 3),
        "min": min(slens) if slens else 0,
        "max": max(slens) if slens else 0,
        "short_ratio": round(sum(1 for x in slens if x < SHORT_SENT) / len(slens), 3) if slens else 0.0,
        "long_ratio": round(sum(1 for x in slens if x > LONG_SENT) / len(slens), 3) if slens else 0.0,
    }
    # 相邻句长变化率：另一种突发性度量
    if len(slens) >= 2:
        diffs = [abs(slens[i + 1] - slens[i]) for i in range(len(slens) - 1)]
        rhythm["jitter"] = round(mean(diffs) / mean(slens), 3) if mean(slens) else 0.0
    else:
        rhythm["jitter"] = 0.0

    # --- 模板词 ---
    hits = []
    for pat, label in TEMPLATE_PATTERNS:
        n = len(re.findall(pat, text))
        if n:
            hits.append({"label": label, "count": n})
    hits.sort(key=lambda h: -h["count"])
    template_count = sum(h["count"] for h in hits)
    template = {
        "total": template_count,
        "density_per_1k": round(template_count / total * 1000, 2) if total else 0.0,
        "hits": hits,
    }

    # --- 标点多样性 ---
    pcount = {}
    for ch in text:
        if ch in PUNCT_KINDS:
            pcount[ch] = pcount.get(ch, 0) + 1
    kinds = len(pcount)
    punct = {
        "kinds": kinds,
        "detail": sorted(pcount.items(), key=lambda kv: -kv[1]),
    }

    # --- 段落 ---
    para = {
        "count": len(plens),
        "mean": round(mean(plens), 1),
        "cv": round(cv(plens), 3) if len(plens) >= 3 else None,
        "lens": plens,
    }

    # --- 词汇 ---
    body = re.sub(r"\s", "", text)
    uniq = len(set(body))
    freq = {}
    for w in FUNCTION_WORDS:
        n = text.count(w)
        if n:
            freq[w] = n
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:8]
    vocab = {
        "chars": total,
        "unique": uniq,
        "richness": round(uniq / len(body), 3) if body else 0.0,
        "top_func": top,
    }

    return {
        "chars": total,
        "sentences": rhythm,
        "template": template,
        "punct": punct,
        "paragraph": para,
        "vocab": vocab,
    }


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def verdict_cv(v):
    if v >= 0.55:
        return "good", "节奏自然"
    if v >= 0.40:
        return "warn", "节奏偏平，还能更起伏"
    return "bad", "节奏过于均匀 —— 典型 AI 特征"


def verdict_short(r):
    if r >= 0.15:
        return "good", "短句占比自然"
    if r >= 0.08:
        return "warn", "短句偏少"
    return "bad", "几乎没有短句 —— 句式单一"


def verdict_long(r):
    if r >= 0.08:
        return "good", "有长句撑起起伏"
    if r >= 0.03:
        return "warn", "长句偏少"
    return "bad", "缺少长句 —— 全是中等长度"


def verdict_template(d):
    if d < 2:
        return "good", "模板词很少"
    if d <= 5:
        return "warn", "模板词偏多"
    return "bad", "模板词密集 —— 一眼 AI"


def verdict_punct(k):
    if k >= 5:
        return "good", "标点丰富"
    if k >= 3:
        return "warn", "标点偏单一"
    return "bad", "标点种类过少"


def verdict_para(c):
    if c is None:
        return "skip", "段落不足 3 段，不判定"
    if c >= 0.35:
        return "good", "段落长度自然"
    if c >= 0.15:
        return "warn", "段落长度偏整齐"
    return "bad", "段落长度高度均匀"


MARK = {"good": "[好]", "warn": "[中]", "bad": "[差]", "skip": "[--]"}


def render_report(m, title="文本 AI 味体检"):
    s, t, p, pa, v = m["sentences"], m["template"], m["punct"], m["paragraph"], m["vocab"]
    L = []
    A = L.append
    A("=" * 64)
    A("  %s（代理指标，非检测器）" % title)
    A("=" * 64)
    A("  字数 %s   句数 %s   段落 %s" % (f"{m['chars']:,}", s["count"], pa["count"]))
    A("")

    g, msg = verdict_cv(s["cv"])
    A("  【句长节奏】%s %s" % (MARK[g], msg))
    A("    平均 %.1f 字   标准差 %.1f   变异系数 %.3f" % (s["mean"], s["stdev"], s["cv"]))
    A("    最短 %d   最长 %d   相邻句长变化率 %.3f" % (s["min"], s["max"], s["jitter"]))
    A("    短句(<%d字) %.1f%%    长句(>%d字) %.1f%%" % (
        SHORT_SENT, s["short_ratio"] * 100, LONG_SENT, s["long_ratio"] * 100))
    A("")

    g, msg = verdict_template(t["density_per_1k"])
    A("  【模板词】%s %s" % (MARK[g], msg))
    A("    命中 %d 处，密度 %.2f / 千字" % (t["total"], t["density_per_1k"]))
    if t["hits"]:
        parts = ["%s×%d" % (h["label"], h["count"]) for h in t["hits"][:10]]
        A("    " + "  ".join(parts))
    else:
        A("    （无命中）")
    A("")

    g, msg = verdict_punct(p["kinds"])
    A("  【标点多样性】%s %s" % (MARK[g], msg))
    A("    使用 %d 种：%s" % (p["kinds"], " ".join(c for c, _ in p["detail"][:14]) or "（无）"))
    A("")

    g, msg = verdict_para(pa["cv"])
    A("  【段落长度】%s %s" % (MARK[g], msg))
    if pa["cv"] is not None:
        A("    %d 段，平均 %.1f 字，变异系数 %.3f" % (pa["count"], pa["mean"], pa["cv"]))
        A("    各段：%s" % " / ".join(str(x) for x in pa["lens"][:14]))
    else:
        A("    仅 %d 段" % pa["count"])
    A("")

    A("  【词汇】")
    A("    不同字符 %d / %d，丰富度 %.3f" % (v["unique"], v["chars"], v["richness"]))
    if v["top_func"]:
        A("    高频虚词：" + "  ".join("%s(%d)" % (w, n) for w, n in v["top_func"]))
    A("")

    # --- 汇总 ---
    scores = {
        "句长节奏": verdict_cv(s["cv"])[0],
        "模板词": verdict_template(t["density_per_1k"])[0],
        "标点多样性": verdict_punct(p["kinds"])[0],
        "段落长度": verdict_para(pa["cv"])[0],
    }
    bad = [k for k, x in scores.items() if x == "bad"]
    warn = [k for k, x in scores.items() if x == "warn"]
    if not bad and not warn:
        overall = "AI 味很淡"
    elif len(bad) >= 2:
        overall = "AI 味偏重"
    elif bad:
        overall = "AI 味明显"
    else:
        overall = "轻微 AI 味"

    A("-" * 64)
    A("  综合判定：%s" % overall)
    if bad:
        A("  优先修：%s" % "、".join(bad))
    if warn:
        A("  可优化：%s" % "、".join(warn))
    if not bad and not warn:
        A("  四项指标均达标。")
    if s["count"] < SMALL_SAMPLE_SENTS or m["chars"] < SMALL_SAMPLE_CHARS:
        A("")
        A("  [注意] 样本偏小（%d 句 / %d 字），比例类指标噪声大，"
          % (s["count"], m["chars"]))
        A("         建议在完整成稿上复测再下结论。")
    A("-" * 64)
    A("  注：代理指标与朱雀/GPTZero 实际打分相关但不等价，")
    A("      参考区间是经验值，建议用真实检测结果校准。")
    return "\n".join(L)


def render_compare(a, b, name_a="原文", name_b="改写后"):
    L = []
    A = L.append
    A("=" * 64)
    A("  指标对比：%s → %s" % (name_a, name_b))
    A("=" * 64)
    rows = [
        ("句长变异系数", a["sentences"]["cv"], b["sentences"]["cv"], True, 0.55),
        ("相邻句长变化率", a["sentences"]["jitter"], b["sentences"]["jitter"], True, 0.50),
        ("短句占比 %", round(a["sentences"]["short_ratio"] * 100, 1),
         round(b["sentences"]["short_ratio"] * 100, 1), True, 15.0),
        ("长句占比 %", round(a["sentences"]["long_ratio"] * 100, 1),
         round(b["sentences"]["long_ratio"] * 100, 1), None, 8.0),
        ("模板词密度 /千字", a["template"]["density_per_1k"],
         b["template"]["density_per_1k"], False, 2.0),
        ("标点种类", a["punct"]["kinds"], b["punct"]["kinds"], True, 5),
        ("段落长度变异系数",
         a["paragraph"]["cv"] if a["paragraph"]["cv"] is not None else 0.0,
         b["paragraph"]["cv"] if b["paragraph"]["cv"] is not None else 0.0, True, 0.35),
        ("字数", a["chars"], b["chars"], None, None),
    ]
    A("  %-20s %10s %10s %8s   %s" % ("指标", name_a, name_b, "变化", "方向"))
    A("  " + "-" * 60)
    better = worse = 0
    for label, va, vb, higher_better, target in rows:
        if higher_better is None:
            delta = "—"
            arrow = "—"
        else:
            d = round(vb - va, 3)
            delta = ("%+.3f" % d) if abs(d) < 10 else ("%+.1f" % d)
            if abs(d) < 1e-9:
                arrow = "持平"
            elif (d > 0) == higher_better:
                arrow = "改善 ↑" if higher_better else "改善 ↓"
                better += 1
            else:
                arrow = "变差 ↓" if higher_better else "变差 ↑"
                worse += 1
        A("  %-20s %10s %10s %8s   %s" % (label, va, vb, delta, arrow))
    A("  " + "-" * 60)

    # 达标情况
    A("")
    A("  达标检查（目标值）：")
    checks = [
        ("句长变异系数 ≥ 0.55", b["sentences"]["cv"] >= 0.55),
        ("模板词密度 < 2/千字", b["template"]["density_per_1k"] < 2),
        ("短句占比 ≥ 15%", b["sentences"]["short_ratio"] >= 0.15),
        ("标点种类 ≥ 5", b["punct"]["kinds"] >= 5),
        ("段落长度 CV ≥ 0.35（≥3段时）",
         b["paragraph"]["cv"] is None or b["paragraph"]["cv"] >= 0.35),
    ]
    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        A("    %s %s" % ("[通过]" if ok else "[未达]", label))
    A("")
    A("  改善 %d 项，变差 %d 项，达标 %d/%d" % (better, worse, passed, len(checks)))
    if passed == len(checks):
        A("  → 可以交付")
    else:
        A("  → 建议针对未达标项再迭代一轮")
    if b["sentences"]["count"] < SMALL_SAMPLE_SENTS or b["chars"] < SMALL_SAMPLE_CHARS:
        A("")
        A("  [注意] 样本偏小（%d 句 / %d 字），比例类指标噪声大。"
          % (b["sentences"]["count"], b["chars"]))
    A("=" * 64)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def read_text(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="文本 AI 味体检（代理指标）")
    ap.add_argument("file", nargs="?", help="文本文件，- 表示 stdin")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="对比两个文件的指标")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--names", nargs=2, default=["原文", "改写后"], help="对比时的名称")
    args = ap.parse_args()

    if args.compare:
        a = analyze(read_text(args.compare[0]))
        b = analyze(read_text(args.compare[1]))
        if args.json:
            print(json.dumps({"before": a, "after": b}, ensure_ascii=False, indent=2))
        else:
            print(render_compare(a, b, args.names[0], args.names[1]))
        return 0

    if not args.file:
        ap.print_help()
        return 1

    m = analyze(read_text(args.file))
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    else:
        print(render_report(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
