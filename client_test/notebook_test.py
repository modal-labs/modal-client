import os
import time
from pathlib import Path
from pprint import pprint

import pytest


@pytest.fixture
def notebook_runner(servicer):
    import papermill

    def runner(notebook_path: Path):
        output_notebook_path = notebook_path.with_name(f"{notebook_path.name}.output.ipynb")

        nb = papermill.execute_notebook(
            notebook_path,
            output_notebook_path,
            parameters={
                "server_addr": servicer.remote_addr,
            }
        )
        for cell in nb["cells"]:
            if cell["metadata"]["papermill"]["exception"]:
                cell_output = cell["outputs"][0]["data"]['text/plain']

                pytest.fail(f"""There was an error when executing the notebook.
Error output:
{cell_output}
Inspect the output notebook: {output_notebook_path}
                """)

    return runner


def test_notebook(notebook_runner, test_dir):
    input_notebook_path = test_dir / "supports" / "notebooks" / "simple.ipynb"
    notebook_runner(input_notebook_path)
