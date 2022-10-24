import os
import time
from pathlib import Path
from pprint import pprint

import pytest


@pytest.fixture
def notebook_runner(servicer):
    import papermill

    def runner(notebook_path: Path):
        output_notebook_path = notebook_path.with_suffix(".output.ipynb")

        nb = papermill.execute_notebook(
            notebook_path,
            output_notebook_path,
            parameters={
                "server_addr": servicer.remote_addr,
            }
        )

        tagged_cells = {}
        for cell in nb["cells"]:
            if cell["metadata"]["papermill"]["exception"]:
                cell_output = cell["outputs"][0]["data"]['text/plain']

                pytest.fail(f"""There was an error when executing the notebook.
Error output:
{cell_output}
Inspect the output notebook: {output_notebook_path}
                """)
            for tag in cell["metadata"]["tags"]:
                tagged_cells[tag] = cell

        return tagged_cells

    return runner


def test_notebook_outputs_status(notebook_runner, test_dir):
    input_notebook_path = test_dir / "supports" / "notebooks" / "simple.ipynb"
    tagged_cells = notebook_runner(input_notebook_path)
    combined_output = '\n'.join(c["data"]["text/plain"] for c in tagged_cells["main"]["outputs"])
    assert "Initialized" in combined_output
    assert "Created objects." in combined_output
    assert "App completed." in combined_output

