import subprocess
import sys


def test_core_cli_help_commands():
    modules = [
        "tacs.data_selection.matching",
        "tacs.data_selection.select_from_scores",
        "tacs.data_selection.write_selected_data",
    ]
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
