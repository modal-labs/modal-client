# Copyright Modal Labs 2022
from pathlib import Path

import jupytext
import pytest
from nbclient.exceptions import CellExecutionError


@pytest.fixture
def notebook_runner(servicer):
    import nbformat
    from nbclient import NotebookClient

    def runner(notebook_path: Path):
        output_notebook_path = notebook_path.with_suffix(".output.ipynb")

        nb = jupytext.read(
            notebook_path,
        )

        parameter_cell = nb["cells"][0]
        assert "parameters" in parameter_cell["metadata"]["tags"]  # like in papermill
        parameter_cell["source"] = f'server_addr = "{servicer.remote_addr}"'

        client = NotebookClient(nb)

        try:
            client.execute()
        except CellExecutionError:
            nbformat.write(nb, output_notebook_path)
            pytest.fail(
                f"""There was an error when executing the notebook.

Inspect the output notebook: {output_notebook_path}
"""
            )
        tagged_cells = {}
        for cell in nb["cells"]:
            for tag in cell["metadata"].get("tags", []):
                tagged_cells[tag] = cell

        return tagged_cells

    return runner


def test_notebook_outputs_status(notebook_runner, test_dir):
    input_notebook_path = test_dir / "supports" / "notebooks" / "simple.notebook.py"
    tagged_cells = notebook_runner(input_notebook_path)
    combined_output = "\n".join(c["data"]["text/plain"] for c in tagged_cells["main"]["outputs"])
    assert "Initialized" in combined_output
    assert "Created objects." in combined_output
    assert "App completed." in combined_output
