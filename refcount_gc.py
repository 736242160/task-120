#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refcount_gc.py — 嵌套对象引用计数回收模拟器（纯标准库单文件）

功能：
  * 载入嵌套对象定义（名称 / 包含的子对象 / 初始外部引用计数）
  * 顺序执行引用操作流：增引用、删引用、建立引用、删除引用
  * 引用计数即时增减，计数归零即时回收，回收时子对象与持有引用级联释放
  * 报告：嵌套循环（包含成环）、循环引用（互相保活）、悬空引用、
          删除不存在的引用、操作已回收对象
  * 回收历史全程可追溯（按操作序号记录原因）

输入格式（文本文件或 stdin，# 开头为注释）：
  obj <名称> <子对象1,子对象2 或 -> <初始引用计数>
  op  <incref|decref|link|unlink> <对象> [目标]
  （op 类型也接受中文：增引用 / 删引用 / 建立引用 / 删除引用）

用法：
  python3 refcount_gc.py 输入文件
  cat 输入文件 | python3 refcount_gc.py
  python3 refcount_gc.py --demo      # 运行内置演示
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------- 数据模型

@dataclass
class GcObject:
    """一个被管理对象。"""
    name: str
    children: list[str] = field(default_factory=list)  # 包含关系（强拥有子对象）
    external: int = 0                                   # 外部引用计数（incref/decref 维护）
    refs: list[str] = field(default_factory=list)       # 本对象建立的引用（可重复，有序）
    count: int = 0                                      # 当前总计数 = 外部 + 被包含 + 被引用
    alive: bool = True


# ---------------------------------------------------------------- 环检测（Tarjan SCC）

def tarjan_scc(nodes, edges):
    """返回图中所有强连通分量。nodes 为节点集合，edges 为 {节点: [邻接节点]}。"""
    sys.setrecursionlimit(max(10000, len(nodes) * 4 + 100))
    index, low, on_stack, stack, sccs = {}, {}, set(), [], []
    counter = [0]

    def strongconnect(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, ()):
            if w not in nodes:
                continue
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in nodes:
        if v not in index:
            strongconnect(v)
    return sccs


def cycle_members(nodes, edges):
    """返回所有处在环上的节点集合（SCC 大小 > 1，或存在自环）。"""
    members = []
    for scc in tarjan_scc(nodes, edges):
        if len(scc) > 1:
            members.append(sorted(scc))
        elif scc[0] in edges.get(scc[0], ()):
            members.append(sorted(scc))
    return members


# ---------------------------------------------------------------- 回收引擎

class Engine:
    def __init__(self):
        self.objects: dict[str, GcObject] = {}
        self.history: list[tuple[int, str, str]] = []   # (操作序号, 对象, 回收原因)
        self.errors: list[tuple[int, str, str]] = []    # (操作序号, 错误类型, 详情)
        self.contain_cycles: list[list[str]] = []       # 嵌套循环（包含关系成环）
        self.seq = 0                                      # 操作序号（0 表示定义阶段）

    # ---- 错误与历史 ----
    def _error(self, kind, detail):
        self.errors.append((self.seq, kind, detail))

    def _record_reclaim(self, obj, reason):
        self.history.append((self.seq, obj.name, reason))

    # ---- 定义阶段 ----
    def add_object(self, name, children, external):
        if name in self.objects:
            self._error("重复定义", f"对象 {name} 被重复定义，后者被忽略")
            return
        self.objects[name] = GcObject(name=name, children=list(children),
                                      external=external)

    def finalize_definitions(self):
        """定义载入完毕：校验子对象存在、初始化计数、检测包含环。"""
        for obj in self.objects.values():
            for ch in obj.children:
                if ch not in self.objects:
                    self._error("未定义子对象",
                                f"对象 {obj.name} 包含未定义的子对象 {ch}")
        # 计数 = 外部引用 + 被包含次数（被引用次数初始为 0）
        for obj in self.objects.values():
            obj.count = obj.external
        for obj in self.objects.values():
            for ch in obj.children:
                if ch in self.objects:
                    self.objects[ch].count += 1
        # 嵌套循环检测（包含关系图上的环）
        edges = {n: [c for c in o.children if c in self.objects]
                 for n, o in self.objects.items()}
        self.contain_cycles = cycle_members(set(self.objects), edges)
        for cyc in self.contain_cycles:
            self._error("嵌套循环",
                        "包含关系成环: " + " -> ".join(cyc + [cyc[0]]))

    # ---- 存活校验 ----
    def _get_alive(self, name, action):
        obj = self.objects.get(name)
        if obj is None:
            self._error("未定义对象", f"{action}: 对象 {name} 未定义")
            return None
        if not obj.alive:
            self._error("操作已回收对象", f"{action}: 对象 {name} 已被回收")
            return None
        return obj

    # ---- 级联回收 ----
    def _reclaim(self, obj, reason):
        """回收对象：释放其子对象与建立的引用，递归级联；检测悬空引用。"""
        if not obj.alive:
            return
        obj.alive = False
        self._record_reclaim(obj, reason)
        # 1) 释放包含的子对象（父对象消亡，子对象失去一份计数）
        for ch in obj.children:
            child = self.objects.get(ch)
            if child is not None and child.alive:
                child.count -= 1
                if child.count <= 0:
                    self._reclaim(child, f"级联回收：父对象 {obj.name} 被回收")
        # 2) 释放本对象建立的引用
        for target_name in obj.refs:
            target = self.objects.get(target_name)
            if target is not None and target.alive:
                target.count -= 1
                if target.count <= 0:
                    self._reclaim(target,
                                  f"级联回收：引用持有者 {obj.name} 被回收")
        obj.refs.clear()
        # 3) 悬空检测：仍有存活对象包含/引用本对象（计数不一致的安全网）
        for other in self.objects.values():
            if other.alive and (obj.name in other.children
                                or obj.name in other.refs):
                self._error("悬空引用",
                            f"对象 {other.name} 仍引用已回收对象 {obj.name}")

    # ---- 操作 ----
    def incref(self, name):
        obj = self._get_alive(name, "增引用")
        if obj is None:
            return
        obj.external += 1
        obj.count += 1

    def decref(self, name):
        obj = self._get_alive(name, "删引用")
        if obj is None:
            return
        if obj.external <= 0:
            self._error("删除不存在的引用",
                        f"对象 {name} 没有可删除的外部引用")
            return
        obj.external -= 1
        obj.count -= 1
        if obj.count <= 0:
            self._reclaim(obj, "外部引用计数归零")

    def link(self, src, dst):
        src_obj = self._get_alive(src, "建立引用")
        dst_obj = self._get_alive(dst, "建立引用")
        if src_obj is None or dst_obj is None:
            return
        src_obj.refs.append(dst)
        dst_obj.count += 1

    def unlink(self, src, dst):
        src_obj = self._get_alive(src, "删除引用")
        if src_obj is None:
            return
        if dst not in src_obj.refs:
            self._error("删除不存在的引用",
                        f"对象 {src} 并不存在指向 {dst} 的引用")
            return
        src_obj.refs.remove(dst)
        dst_obj = self.objects.get(dst)
        if dst_obj is not None and dst_obj.alive:
            dst_obj.count -= 1
            if dst_obj.count <= 0:
                self._reclaim(dst_obj,
                              f"引用 {src} -> {dst} 被删除后计数归零")

    # ---- 终态分析 ----
    def detect_reference_cycles(self):
        """在存活对象的 包含+引用 合并图上检测互相保活的环。"""
        alive = {n for n, o in self.objects.items() if o.alive}
        edges = {}
        for n in alive:
            o = self.objects[n]
            edges[n] = ([c for c in o.children if c in alive]
                        + [r for r in o.refs if r in alive])
        reports = []
        for cyc in cycle_members(alive, edges):
            has_external = any(self.objects[n].external > 0 for n in cyc)
            reports.append((cyc, has_external))
        return reports

    # ---- 报告 ----
    def report(self, out=sys.stdout):
        p = lambda *a: print(*a, file=out)
        p("=" * 60)
        p("回收历史（按发生顺序，可追溯）")
        p("=" * 60)
        if not self.history:
            p("  （无对象被回收）")
        for i, (seq, name, reason) in enumerate(self.history, 1):
            p(f"  #{i:<3} [操作序号 {seq:>3}] 回收 {name:<12} 原因: {reason}")

        p()
        p("=" * 60)
        p("错误与警告清单")
        p("=" * 60)
        if not self.errors:
            p("  （无错误）")
        for seq, kind, detail in self.errors:
            stage = "定义阶段" if seq == 0 else f"操作序号 {seq}"
            p(f"  [{stage:>9}] {kind}: {detail}")

        p()
        p("=" * 60)
        p("循环引用（互相保活）检测报告")
        p("=" * 60)
        cycles = self.detect_reference_cycles()
        if not cycles:
            p("  （未检测到存活对象间的环）")
        for cyc, has_external in cycles:
            tag = "环上有外部引用，断开外部引用后仍无法回收" if has_external \
                  else "环内互相保活，已无法从外部到达（内存泄漏）"
            p(f"  环: {' -> '.join(cyc + [cyc[0]])}  -- {tag}")

        p()
        p("=" * 60)
        p("终态快照")
        p("=" * 60)
        alive = [o for o in self.objects.values() if o.alive]
        dead = [o for o in self.objects.values() if not o.alive]
        p(f"  存活对象 {len(alive)} 个: "
          + (", ".join(f"{o.name}(count={o.count}, external={o.external})"
                       for o in alive) or "无"))
        p(f"  已回收   {len(dead)} 个: "
          + (", ".join(o.name for o in dead) or "无"))


# ---------------------------------------------------------------- 输入解析

OP_ALIASES = {
    "incref": "incref", "增引用": "incref",
    "decref": "decref", "删引用": "decref",
    "link": "link", "建立引用": "link",
    "unlink": "unlink", "删除引用": "unlink",
}


def run_script(engine: Engine, lines):
    """解析并执行定义与操作流。定义必须先于操作出现。"""
    in_ops = False
    for lineno, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        head = parts[0].lower()
        if head == "obj":
            if in_ops:
                engine._error("格式错误",
                              f"第 {lineno} 行: 对象定义必须位于操作流之前")
                continue
            if len(parts) != 4:
                engine._error("格式错误",
                              f"第 {lineno} 行: 应为 obj <名称> <子对象|-> <计数>")
                continue
            _, name, children_s, count_s = parts
            try:
                count = int(count_s)
            except ValueError:
                engine._error("格式错误", f"第 {lineno} 行: 计数 {count_s} 非整数")
                continue
            if count < 0:
                engine._error("格式错误", f"第 {lineno} 行: 计数不能为负")
                continue
            children = [] if children_s == "-" else [
                c for c in children_s.split(",") if c]
            engine.add_object(name, children, count)
        elif head == "op":
            if not in_ops:
                in_ops = True
                engine.finalize_definitions()
            if len(parts) < 3:
                engine._error("格式错误", f"第 {lineno} 行: 操作参数不足")
                continue
            op = OP_ALIASES.get(parts[1].lower()) or OP_ALIASES.get(parts[1])
            if op is None:
                engine._error("格式错误", f"第 {lineno} 行: 未知操作 {parts[1]}")
                continue
            engine.seq += 1
            if op in ("incref", "decref"):
                if len(parts) != 3:
                    engine._error("格式错误",
                                  f"第 {lineno} 行: {op} 只需一个对象参数")
                    continue
                getattr(engine, op)(parts[2])
            else:
                if len(parts) != 4:
                    engine._error("格式错误",
                                  f"第 {lineno} 行: {op} 需要 对象 与 目标")
                    continue
                getattr(engine, op)(parts[2], parts[3])
        else:
            engine._error("格式错误", f"第 {lineno} 行: 未知指令 {parts[0]}")
    if not in_ops:
        engine.finalize_definitions()


# ---------------------------------------------------------------- 内置演示

DEMO = """\
# 所有对象定义先于操作流给出；操作按场景分组
# ---- 场景 1 用到的对象：root 包含 a、b，b 包含 c；log 为独立对象 ----
obj root   a,b   1
obj a      -     0
obj b      c     0
obj c      -     0
obj log    -     0
# ---- 场景 2 用到的对象：x 包含 y、y 包含 x（嵌套循环） ----
obj x      y     1
obj y      x     0
# ---- 场景 3 用到的对象：p、q 互相引用（互相保活） ----
obj p      -     1
obj q      -     1
# ---- 场景 4 用到的对象 ----
obj r      -     1

# 场景 1：级联回收
op link root log
op decref root          # root 归零 -> 级联回收 a、b、c、log
op incref a             # 错误：操作已回收对象
op decref root          # 错误：操作已回收对象

# 场景 3：循环引用（嵌套循环 x/y 在定义阶段已报告）
op link p q
op link q p
op decref p             # p 仍被 q 引用，存活
op decref q             # q 仍被 p 引用，存活 -> 终态报告环

# 场景 4：删除不存在的引用
op unlink r q           # 错误：r 没有指向 q 的引用
op decref r
op decref r             # 错误：r 已被回收（且无可删外部引用）
"""


# ---------------------------------------------------------------- 入口

def main(argv=None):
    ap = argparse.ArgumentParser(description="嵌套对象引用计数回收模拟器")
    ap.add_argument("input", nargs="?", help="输入文件（缺省读 stdin）")
    ap.add_argument("--demo", action="store_true", help="运行内置演示")
    args = ap.parse_args(argv)

    if args.demo:
        lines = DEMO.splitlines()
    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            lines = f.read().splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    engine = Engine()
    run_script(engine, lines)
    engine.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
