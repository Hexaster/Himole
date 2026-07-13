"""Keep executable commands in the Build-2 map synchronized with the repository."""
import ast
import re
import shlex
from pathlib import Path

from scripts.compare_baseline_himole import parse_args as parse_compare_args
from scripts.train_baseline import parse_args as parse_baseline_args
from scripts.train_himole import parse_args as parse_himole_args


ROOT = Path(__file__).resolve().parents[1]
MAP_TEXT = (ROOT / "docs" / "build2-map.html").read_text(encoding="utf-8")
COMMANDS = [
    line.strip()
    for block in re.findall(r"<pre>(.*?)</pre>", MAP_TEXT, flags=re.DOTALL)
    for line in block.splitlines()
    if line.strip().startswith(("python ", "pytest "))
]


def test_every_documented_python_command_has_a_script_and_valid_arguments():
    parsers = {
        "scripts/train_baseline.py": parse_baseline_args,
        "scripts/train_himole.py": parse_himole_args,
        "scripts/compare_baseline_himole.py": parse_compare_args,
    }

    for command in (command for command in COMMANDS if command.startswith("python ")):
        tokens = shlex.split(command)
        script = tokens[1]
        assert (ROOT / script).is_file(), command
        assert script in parsers, command
        parsers[script](tokens[2:])


def test_every_documented_pytest_target_exists():
    for command in (command for command in COMMANDS if command.startswith("pytest ")):
        for token in shlex.split(command)[1:]:
            if token.startswith("-") or ".py" not in token:
                continue
            file_name, _, node_name = token.partition("::")
            path = ROOT / file_name
            assert path.is_file(), command
            if node_name:
                functions = {
                    node.name
                    for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                assert node_name in functions, command
