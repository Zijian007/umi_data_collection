"""当前操作示例、导出字段和代码指南的轻量契约检查。"""

import ast
from pathlib import Path
import re
import shlex
import tempfile
import unittest

from software.infrastructure.file_store import write_yaml
from software.scene.cli import build_parser
from software.scene.overrides import SceneOverrides, apply_overrides


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = ('software/scene/README.md', 'software/trajectory/README.md',
                'software/dataset/README.md', 'exports/README.md', 'docs/场景处理.md',
                'docs/训练数据.md', 'docs/命令行参数.md')


class DocumentationContractsTests(unittest.TestCase):
    def test_current_document_links_resolve(self):
        for name in CURRENT_DOCS:
            path = ROOT / name
            for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text()):
                if target.startswith(('#', 'http:', 'https:', 'codex:')):
                    continue
                target = target.split('#', 1)[0]
                with self.subTest(document=name, target=target):
                    self.assertTrue((path.parent / target).exists())

    def test_marked_examples_use_production_config_and_cli_parsers(self):
        from ruamel.yaml import YAML
        counts = dict.fromkeys(('scene-expert', 'scene-export'), 0)
        for name in CURRENT_DOCS:
            for kind, language, source in re.findall(
                    r'<!-- check: ([\w-]+) -->\s*```(\w+)\n(.*?)```', (ROOT / name).read_text(), re.S):
                with self.subTest(document=name, kind=kind):
                    self.assertIn(kind, counts)
                    counts[kind] += 1
                    if kind == 'scene-expert':
                        self.assertEqual(language, 'yaml')
                        document = YAML(typ='safe').load(source)
                        # 使用正常加载入口，无副本 schema 或模拟配置解析器。
                        with tempfile.TemporaryDirectory() as temporary:
                            path = Path(temporary) / 'expert.yaml'
                            write_yaml(path, document)
                            from software.configuration.loader import load
                            from software.scene.management.inputs.scene import load_scene_parameters
                            apply_overrides(load('trajectory'), load_scene_parameters(), SceneOverrides.load(path))
                    else:
                        command = shlex.split(source.replace('\\\n', ' '))
                        self.assertEqual(command.pop(0), './umi-scene')
                        parsed = build_parser().parse_args(command)
                        self.assertEqual(parsed.command, 'export')
                        self.assertEqual(parsed.format, 'training')
        self.assertTrue(all(counts.values()))

    def test_scene_call_chain_references_existing_functions(self):
        # 指南的明确阅读入口必须存在；解析 AST，不导入或启动求解器。
        references = {
            'pipeline/tasks/estimation.py': ('select_map', 'execute'),
            'pipeline/tasks/backends/scene_map.py': ('execute_task',),
            'pipeline/tasks/results/arm_writer.py': ('write_arm_result',),
            'pipeline/tasks/results/trajectory_writers.py': ('write_bimanual_trajectory',),
            'pipeline/tasks/results/result_writer.py': ('write_episode_result',),
            'management/batch/snapshot.py': ('read_inputs', 'make_record', 'save_record'),
        }
        guide = (ROOT / 'software/scene/README.md').read_text()
        for module, functions in references.items():
            tree = ast.parse((ROOT / 'software/scene' / module).read_text())
            defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            for function in functions:
                with self.subTest(module=module, function=function):
                    self.assertIn(function, defined)
                    self.assertIn(function, guide)
