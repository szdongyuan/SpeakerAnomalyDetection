"""Semantic migration guard; real routing is covered by business logging tests."""
import ast
from pathlib import Path
import subprocess

import pytest


def logging_bypasses(source):
    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'logging':
                    aliases[alias.asname or alias.name] = 'logging'
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'logging':
            for alias in node.names:
                aliases[alias.asname or alias.name] = 'logging.' + alias.name

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    # A finite fixed point handles chains of simple aliases independent of nesting.
    for _ in range(len(assignments) + 1):
        before = dict(aliases)
        for node in assignments:
            value = resolve(node.value)
            if value:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = value
        if before == aliases:
            break
    output = {'debug', 'info', 'warning', 'warn', 'error', 'exception', 'critical', 'fatal', 'log'}
    forbidden = {'logging.' + name for name in output | {'getLogger', 'basicConfig'}}
    forbidden |= {'logging.root.' + name for name in output}
    return [(node.lineno, resolve(node.func)) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and resolve(node.func) in forbidden]


@pytest.mark.parametrize('source', [
    'import logging; logging.getLogger("business")',
    'import logging as log; log.warning("business")',
    'from logging import getLogger as acquire; acquire("business")',
    'from logging import basicConfig as configure; configure()',
    'from logging import error; error("business")',
    'import logging as log; root = log.root; root.info("business")',
    'import logging; acquire = logging.getLogger; acquire("business")',
    'import logging; log = logging; log.exception("business")',
])
def test_guard_detects_raw_calls_and_aliases(source):
    assert logging_bypasses(source)


@pytest.mark.parametrize('source', [
    'import logging; level = logging.INFO',
    'from logging import LoggerAdapter; adapter = LoggerAdapter(project, {})',
    'import logging as log; adapter = log.LoggerAdapter(project, {})',
    'from logging import Logger; logger: Logger = project; logger.info("ok")',
    'from base.log_manager import LogManager; LogManager.set_log_handler("core").info("ok")',
    'import logging; text = "logging.getLogger()"',
])
def test_guard_allows_levels_types_adapters_and_project_routes(source):
    assert logging_bypasses(source) == []


def production_logging_violations(root):
    paths = subprocess.run(['git', 'ls-files', '-z', '--', '*.py'], cwd=root,
                           capture_output=True, check=True, timeout=30).stdout.decode().split('\0')
    violations = []
    for name in paths:
        path = Path(name)
        if not name or name == 'base/log_manager.py':
            continue
        # Runtime roots include every project-owned package and both launchers.
        # Tests, scripts/tools, dependencies/build output and worktrees are not runtime.
        if len(path.parts) > 1 and path.parts[0] not in {'base', 'consts', 'ui'}:
            continue
        source = (root / path).read_text(encoding='utf-8-sig')
        violations.extend(f'{name}:{line}: {call}' for line, call in logging_bypasses(source))
    return violations


def test_production_logging_uses_project_entrypoint():
    assert production_logging_violations(Path(__file__).resolve().parents[2]) == []
