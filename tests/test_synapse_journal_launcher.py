"""CPU safety tests for the independent journal launcher; all state is temporary."""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_synapse_journal_ablation as launcher






class JournalLauncherFixtures:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "outputs/ablations/journal_20261025"
        self.base.mkdir(parents=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.temporary.cleanup)
        self.stack.enter_context(patch.object(launcher, "ROOT", self.root))
        self.stack.enter_context(patch.object(launcher, "BASE", self.base))


class JournalSharedUtilityTests(unittest.TestCase):
    def test_environment_removes_hidden_gpu_without_mutating_parent(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "0"}):
            actual = launcher.environment()
            self.assertNotIn("CUDA_VISIBLE_DEVICES", actual)
            self.assertEqual(actual["HF_HUB_OFFLINE"], "1")
            self.assertEqual(actual["TORCH_HOME"], "/Disk/HDD/Sk/.cache/torch")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")


class JournalPlanStatusTests(JournalLauncherFixtures, unittest.TestCase):
    def plan_path(self):
        path = self.root / ".codex/synapse_journal_ablation_plan.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_only_section11_status_row_changes_other_bytes_preserved(self):
        old = """# 论文消融
## 2. 主模型对照
| ID | 模型 | 训练增强 | 论文推理 | 工作内容 |
|---|---|---|---|---|
| M0 | baseline | 原增强 | 1-view | 补测 |
| M1 | baseline+ASEE | 原增强 | 1-view | 补测 |
## 4. ASEE机制
| ID | 因素 | 解释 |
|---|---|---|
| M1 | 双分支 | 机制对照 |
## 11. 实验状态登记
| 顺序 | ID | 内容 | 状态 | 结果与证据 |
|---|---|---|---|---|
| 1 | M0 | baseline已有best + 1-view | 已完成 | old_m0.md |
| 2 | M1 | baseline+ASEE已有best + 1-view | 待执行 | — |
| 3 | M10 | 相似ID不能修改 | 待执行 | — |
## 12. 后文
| 2 | M1 | 同格式其他节 | 待执行 | — |
"""
        path = self.plan_path()
        path.write_text(old, encoding="utf-8")
        launcher.set_plan_status("M1", "已完成", "new_m1.md")
        expected = old.replace(
            "| 2 | M1 | baseline+ASEE已有best + 1-view | 待执行 | — |",
            "| 2 | M1 | baseline+ASEE已有best + 1-view | 已完成 | new_m1.md |")
        self.assertEqual(path.read_bytes(), expected.encode("utf-8"))

    def test_missing_section11_target_raises_without_writing(self):
        old = """# 论文消融
## 2. 主模型对照
| M1 | 原表不该匹配 | 固定1-view |
## 4. ASEE机制
| M1 | 机制行不该匹配 | 原描述 |
## 11. 实验状态登记
| 顺序 | ID | 内容 | 状态 | 结果与证据 |
| 1 | M0 | baseline | 已完成 | evidence |
## 12. 后文
| 2 | M1 | 其他节五列表 | 待执行 | — |
"""
        path = self.plan_path()
        path.write_text(old, encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "section 11"):
            launcher.set_plan_status("M1", "已完成", "must_not_write.md")
        self.assertEqual(path.read_bytes(), before)






if __name__ == "__main__":
    unittest.main()
