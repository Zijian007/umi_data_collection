"""当前操作示例、导出字段和代码指南的轻量契约检查。"""

import ast
from pathlib import Path
import re
import shlex
import tempfile
import unittest

from software.infrastructure.file_store import write_yaml
from software.scene.cli import build_parser
from software.scene.parameters import SceneOverrides, apply_overrides


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    'README.md', 'config/README.md', 'exports/README.md', 'kits/README.md', 'scenes/README.md',
    *(str(path.relative_to(ROOT)) for path in sorted((ROOT / 'docs').glob('*.md'))),
    'docs/dev/README.md', 'docs/dev/SSOT与权威来源.md', 'docs/dev/算法实现约定.md',
    'docs/dev/标定架构与验证.md',
    'software/scene/README.md', 'software/trajectory/README.md',
    'software/trajectory/algorithms/poster_vio/README.md', 'software/dataset/README.md', 'software/vision/README.md',
)


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
        counts = dict.fromkeys(('scene-expert', 'scene-export', 'collection'), 0)
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
                    elif kind == 'collection':
                        from software.scene.management.collection_schema import normalize_collection
                        normalize_collection(YAML(typ='safe').load(source))
                    else:
                        command = shlex.split(source.replace('\\\n', ' '))
                        self.assertEqual(command.pop(0), './umi-scene')
                        parsed = build_parser().parse_args(command)
                        self.assertEqual(parsed.command, 'export')
                        self.assertEqual(parsed.format, 'training')
        self.assertTrue(all(counts.values()))

    def test_collection_template_uses_the_public_schema(self):
        from ruamel.yaml import YAML
        from software.scene.management.collection_schema import normalize_collection
        source = (ROOT / 'scenes/example_scene/collection.example.yaml').read_text()
        normalize_collection(YAML(typ='safe').load(source))

    def test_yaml_scalar_keys_have_user_facing_comments(self):
        # YAML defaults and examples are user-facing; every scalar declaration
        # must state its unit, candidate values, default or special-value rule.
        import re
        paths = [* (ROOT / 'config').rglob('*.yaml'),
                 * (ROOT / 'kits').glob('*/kit.yaml'),
                 * (ROOT / 'scenes').rglob('collection.yaml'),
                 ROOT / 'scenes/example_scene/collection.example.yaml']
        for path in paths:
            if '.pipeline' in path.parts or path.name.endswith('.local.yaml'):
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                match = re.match(r'^\s*[A-Za-z_][\w-]*:(?:\s*(.*))?$', line)
                if match and (match.group(1) or '').strip() and '#' not in line:
                    self.fail(f'{path}:{number} scalar YAML key lacks a comment: {line.strip()}')

    def test_scene_call_chain_references_existing_functions(self):
        # 指南的明确阅读入口必须存在；解析 AST，不导入或启动求解器。
        references = {
            'pipeline/tasks/backend_routing.py': ('select_map', 'execute'),
            'pipeline/tasks/scene_map.py': ('execute_task',),
            'pipeline/tasks/trajectory_writers.py': ('write_arm_result', 'write_bimanual_trajectory'),
            'pipeline/tasks/publication.py': ('write_episode_result',),
            'management/batch_snapshot.py': ('read_inputs', 'make_record', 'save_record'),
        }
        guide = (ROOT / 'software/scene/README.md').read_text()
        for module, functions in references.items():
            tree = ast.parse((ROOT / 'software/scene' / module).read_text())
            defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            for function in functions:
                with self.subTest(module=module, function=function):
                    self.assertIn(function, defined)
                    self.assertIn(function, guide)
