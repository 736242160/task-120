#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_gc.py — 嵌套对象引用计数回收模拟器（纯 Python 标准库，单文件）

输入（JSON，从文件或 stdin 读取）：
{
  "objects": [
    {"name": "a", "children": ["b", "c"], "refcount": 1},
    ...
  ],
  "operations": [
    {"op": "incref", "object": "a"},                 # 增引用
    {"op": "decref", "object": "a"},                 # 删引用
    {"op": "link", "object": "a", "target": "b"}     # 建立引用 a -> b
  ]
}

语义模型：
- 每个对象有引用计数；incref/decref 立即增减计数，link 使目标计数 +1 并记录有向边。
- 包含关系（children）是所有权关系：定义时每个子对象因被包含而 +1；
  父对象回收时对子对象做级联 decref，归零即级联回收。
- 计数归零的对象立即回收；回收时释放其全部出边（link 引用）与子对象（级联）。
- 回收后仍被存活对象引用的，报告悬空引用；对已回收对象的任何操作报错。

输出：回收历史（可追溯）、对象最终状态、错误与报告清单
（嵌套包含环 / 循环引用保活 / 悬空引用 / 删除不存在的引用 / 操作已回收对象等）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field


@dataclass
class Obj:
    name: str
    children: list          # 包含的子对象名（所有权，贡献 +1 计数）
    refcount: int
    alive: bool = True
    refs_out: list = field(default_factory=list)  # link 建立的出边（可重复）
    refs_in: list = field(default_factory=list)   # link 建立的入边


class Engine:
    """引用计数回收引擎：即时增减、即时回收、级联联动、全程留痕。"""

    def __init__(self):
        self.objects = {}       # name -> Obj
        self.errors = []        # 错误/报告清单（有序）
        self.history = []       # 回收历史（有序、可追溯）
        self._seq = 0
        self._op_index = None   # 当前执行的操作下标（用于错误定位）

    # ---------- 记录辅助 ----------

    def _error(self, kind, message, **extra):
        entry = {"kind": kind, "message": message}
        if self._op_index is not None:
            entry["op_index"] = self._op_index
        entry.update(extra)
        self.errors.append(entry)

    def _log_reclaim(self, name, reason, depth):
        self._seq += 1
        self.history.append({
            "seq": self._seq,
            "object": name,
            "reason": reason,
            "cascade_depth": depth,
        })

    # ---------- 装载对象定义 ----------

    def load(self, defs):
        for d in defs:
            name = d.get("name")
            if not name:
                self._error("invalid_definition", "对象定义缺少 name 字段", definition=d)
                continue
            if name in self.objects:
                self._error("duplicate_definition", f"对象重复定义: {name}", object=name)
                continue
            self.objects[name] = Obj(
                name=name,
                children=list(d.get("children", [])),
                refcount=int(d.get("refcount", 0)),
            )
        # 校验子对象存在性
        for o in self.objects.values():
            for c in o.children:
                if c not in self.objects:
                    self._error("unknown_child",
                                f"对象 {o.name} 包含未定义的子对象 {c}",
                                object=o.name, child=c)
        # 包含关系是所有权：被包含即 +1（父回收时级联 decref 的对应项）
        for o in self.objects.values():
            for c in o.children:
                child = self.objects.get(c)
                if child is not None:
                    child.refcount += 1
        # 嵌套包含成环检测（a 包含 b、b 包含 a）
        contain_graph = {n: [c for c in o.children if c in self.objects]
                         for n, o in self.objects.items()}
        for cyc in self._find_cycles(contain_graph):
            self._error("containment_cycle",
                        "嵌套包含成环: " + " -> ".join(cyc + [cyc[0]]),
                        cycle=cyc)
        # 初始计数为 0 的对象立即回收（级联可能连锁触发）
        for o in list(self.objects.values()):
            if o.alive and o.refcount <= 0:
                self._reclaim(o, "初始引用计数为 0", 0)

    # ---------- 引用计数核心 ----------

    def incref(self, name):
        o = self.objects.get(name)
        if o is None:
            self._error("unknown_object", f"增引用失败，对象不存在: {name}", object=name)
            return
        if not o.alive:
            self._error("use_after_free",
                        f"增引用失败，对象已回收: {name}", object=name)
            return
        o.refcount += 1

    def decref(self, name):
        o = self.objects.get(name)
        if o is None:
            self._error("unknown_object", f"删引用失败，对象不存在: {name}", object=name)
            return
        if not o.alive:
            self._error("missing_reference",
                        f"删除不存在的引用: 对象 {name} 已被回收", object=name)
            return
        if o.refcount <= 0:  # 防御性分支：正常模型下存活对象计数恒 >= 1
            self._error("missing_reference",
                        f"删除不存在的引用: 对象 {name} 当前引用计数为 0", object=name)
            return
        o.refcount -= 1
        if o.refcount == 0:
            self._reclaim(o, "显式删引用导致计数归零", 0)

    def link(self, src, tgt):
        ok = True
        so = self.objects.get(src)
        to = self.objects.get(tgt)
        if so is None:
            self._error("unknown_object", f"建立引用失败，源对象不存在: {src}", object=src)
            ok = False
        elif not so.alive:
            self._error("use_after_free",
                        f"建立引用失败，源对象已回收: {src}", object=src)
            ok = False
        if to is None:
            self._error("unknown_object", f"建立引用失败，目标对象不存在: {tgt}", target=tgt)
            ok = False
        elif not to.alive:
            self._error("use_after_free",
                        f"建立引用失败，目标对象已回收: {tgt}", target=tgt)
            ok = False
        if not ok:
            return
        so.refs_out.append(tgt)
        to.refs_in.append(src)
        to.refcount += 1

    # ---------- 回收与级联 ----------

    def _decref_internal(self, name, reason, depth):
        """级联路径上的内部 decref：命中已回收对象时静默跳过（如包含环）。"""
        o = self.objects.get(name)
        if o is None or not o.alive:
            return
        o.refcount -= 1
        if o.refcount <= 0:
            o.refcount = 0
            self._reclaim(o, reason, depth)

    def _reclaim(self, o, reason, depth):
        if not o.alive:
            return
        o.alive = False
        o.refcount = 0
        self._log_reclaim(o.name, reason, depth)
        # 悬空检测：回收瞬间仍有存活对象持有指向它的引用
        for src in list(o.refs_in):
            so = self.objects.get(src)
            if so is not None and so.alive:
                self._error("dangling_reference",
                            f"悬空引用: 存活对象 {src} 仍引用已回收对象 {o.name}",
                            object=src, target=o.name)
        # 级联 1：释放全部出边引用（link 建立的对目标的保活）
        for tgt in o.refs_out:
            to = self.objects.get(tgt)
            if to is not None and o.name in to.refs_in:
                to.refs_in.remove(o.name)
            self._decref_internal(tgt, f"引用源 {o.name} 被回收，释放其出边引用", depth + 1)
        o.refs_out.clear()
        o.refs_in.clear()
        # 级联 2：释放包含的子对象（所有权联动）
        for c in o.children:
            self._decref_internal(c, f"父对象 {o.name} 被回收，子对象级联释放", depth + 1)

    # ---------- 操作流执行 ----------

    def run(self, ops):
        for i, op in enumerate(ops):
            self._op_index = i
            kind = op.get("op")
            if kind == "incref":
                self.incref(op.get("object"))
            elif kind == "decref":
                self.decref(op.get("object"))
            elif kind == "link":
                self.link(op.get("object"), op.get("target"))
            else:
                self._error("unknown_operation", f"未知操作类型: {kind!r}", operation=op)
        self._op_index = None

    # ---------- 环检测（Tarjan SCC，迭代实现） ----------

    @staticmethod
    def _find_cycles(graph):
        """返回图中所有环（大小>1 的强连通分量，或自环），每环为排序后的节点列表。"""
        index_of, low, on_stack, stack = {}, {}, set(), []
        cycles = []
        counter = [0]
        for root in graph:
            if root in index_of:
                continue
            index_of[root] = low[root] = counter[0]
            counter[0] += 1
            stack.append(root)
            on_stack.add(root)
            work = [(root, iter(graph[root]))]
            while work:
                node, it = work[-1]
                descended = False
                for nxt in it:
                    if nxt not in graph:
                        continue
                    if nxt not in index_of:
                        index_of[nxt] = low[nxt] = counter[0]
                        counter[0] += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append((nxt, iter(graph[nxt])))
                        descended = True
                        break
                    elif nxt in on_stack:
                        low[node] = min(low[node], index_of[nxt])
                if descended:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index_of[node]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(w)
                        if w == node:
                            break
                    if len(scc) > 1 or scc[0] in graph[scc[0]]:
                        cycles.append(sorted(scc))
        return cycles

    # ---------- 汇总报告 ----------

    def finalize(self):
        # 循环引用检测：存活对象间的 link 边成环（互相保活，计数永远归不了零）
        live_graph = {
            n: [t for t in o.refs_out if self.objects[t].alive]
            for n, o in self.objects.items() if o.alive
        }
        for cyc in self._find_cycles(live_graph):
            self._error("reference_cycle",
                        "循环引用互相保活: " + " -> ".join(cyc + [cyc[0]]),
                        cycle=cyc)
        # 悬空兜底扫描（回收时刻已报告过的不再重复）
        seen = {(e.get("object"), e.get("target"))
                for e in self.errors if e["kind"] == "dangling_reference"}
        for n, o in self.objects.items():
            if not o.alive:
                continue
            for t in o.refs_out:
                if not self.objects[t].alive and (n, t) not in seen:
                    self._error("dangling_reference",
                                f"悬空引用: 存活对象 {n} 仍引用已回收对象 {t}",
                                object=n, target=t)
                    seen.add((n, t))

    def report(self):
        self.finalize()
        alive = sorted(n for n, o in self.objects.items() if o.alive)
        reclaimed = [h["object"] for h in self.history]
        return {
            "reclaim_history": self.history,
            "objects": {
                n: {
                    "alive": o.alive,
                    "refcount": o.refcount,
                    "children": o.children,
                    "refs_out": list(o.refs_out),
                } for n, o in sorted(self.objects.items())
            },
            "alive": alive,
            "reclaimed": reclaimed,
            "errors": self.errors,
            "summary": {
                "total_objects": len(self.objects),
                "alive_count": len(alive),
                "reclaimed_count": len(reclaimed),
                "error_count": len(self.errors),
            },
        }


# ---------- 输入/输出 ----------

def print_text(rep):
    print("== 回收历史 ==")
    if not rep["reclaim_history"]:
        print("  (无对象被回收)")
    for h in rep["reclaim_history"]:
        print(f"  #{h['seq']} 回收 {h['object']}  "
              f"[级联深度 {h['cascade_depth']}] 原因: {h['reason']}")
    print("\n== 对象最终状态 ==")
    for n, st in rep["objects"].items():
        status = "存活" if st["alive"] else "已回收"
        refs = ",".join(st["refs_out"]) or "-"
        children = ",".join(st["children"]) or "-"
        print(f"  {n}: {status}  refcount={st['refcount']}  "
              f"children=[{children}]  refs_out=[{refs}]")
    print("\n== 错误与报告清单 ==")
    if not rep["errors"]:
        print("  (无错误)")
    for i, e in enumerate(rep["errors"], 1):
        loc = f"op#{e['op_index']} " if "op_index" in e else ""
        print(f"  {i}. [{e['kind']}] {loc}{e['message']}")
    s = rep["summary"]
    print(f"\n== 汇总 ==  对象 {s['total_objects']} 个，存活 {s['alive_count']}，"
          f"回收 {s['reclaimed_count']}，错误/报告 {s['error_count']} 条")


DEMO_INPUT = {
    "objects": [
        {"name": "root", "children": ["a"], "refcount": 1},
        {"name": "a", "children": ["b"], "refcount": 0},
        {"name": "b", "children": [], "refcount": 0},
        {"name": "x", "children": ["y"], "refcount": 1},
        {"name": "y", "children": ["x"], "refcount": 0},
        {"name": "p", "children": [], "refcount": 1},
        {"name": "q", "children": [], "refcount": 1},
        {"name": "m", "children": [], "refcount": 1},
        {"name": "n", "children": [], "refcount": 1},
    ],
    "operations": [
        {"op": "link", "object": "p", "target": "q"},    # q: 1 -> 2
        {"op": "decref", "object": "q"},                 # q: 2 -> 1
        {"op": "decref", "object": "q"},                 # q 归零回收，p 仍引用 q -> 悬空
        {"op": "link", "object": "q", "target": "p"},    # 源已回收 -> 报错
        {"op": "decref", "object": "q"},                 # 删除不存在的引用 -> 报错
        {"op": "decref", "object": "root"},              # root 回收，级联 a -> b
        {"op": "incref", "object": "b"},                 # 操作已回收对象 -> 报错
        {"op": "link", "object": "p", "target": "a"},    # 目标已回收 -> 报错
        {"op": "link", "object": "m", "target": "n"},    # m <-> n 循环引用互相保活
        {"op": "link", "object": "n", "target": "m"},
    ],
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="嵌套对象引用计数回收模拟器（纯标准库单文件）")
    ap.add_argument("input", nargs="?",
                    help="输入 JSON 文件；缺省读 stdin；用 --demo 跑内置示例")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出完整报告")
    ap.add_argument("--demo", action="store_true", help="运行内置演示用例")
    args = ap.parse_args(argv)

    if args.demo:
        data = DEMO_INPUT
    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    eng = Engine()
    eng.load(data.get("objects", []))
    eng.run(data.get("operations", []))
    rep = eng.report()

    if args.json:
        json.dump(rep, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print_text(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
