# Copyright Modal Labs 2022
import pytest
from pathlib import Path

import jupytext
from nbclient.exceptions import CellExecutionError


@pytest.fixture
def notebook_runner(servicer, credentials):
    import nbformat
    from nbclient import NotebookClient

    def runner(notebook_path: Path):
        output_notebook_path = notebook_path.with_suffix(".output.ipynb")

        nb = jupytext.read(
            notebook_path,
        )

        parameter_cell = nb["cells"][0]
        assert "parameters" in parameter_cell["metadata"]["tags"]  # like in papermill
        parameter_cell["source"] = f'''
server_addr = "{servicer.client_addr}"
token_id = "{credentials[0]}"
token_secret = "{credentials[1]}"
'''

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


# for some reason this import is failing due to a circular import of IPython.terminal.embed
# but only when running in CI (sometimes?), causing these tests to fail:
# from IPython.terminal import interactiveshell


# @pytest.mark.skip("temporarily disabled until IPython import issues in CI are resolved")
def test_notebook_outputs_status(notebook_runner, test_dir):
    input_notebook_path = test_dir / "supports" / "notebooks" / "simple.notebook.py"
    tagged_cells = notebook_runner(input_notebook_path)
    combined_output = "\n".join(output_part["text"] for output_part in tagged_cells["main"]["outputs"])
    assert "Initialized" in combined_output
    assert "Created objects." in combined_output
    assert "App completed." in combined_output


def test_is_interactive_ipython_in_real_notebook(notebook_runner, test_dir):
    """Integration test: Run actual notebook to verify is_interactive_ipython returns True."""
    notebook_path = test_dir / "supports" / "notebooks" / "ipython_detection.notebook.py"
    tagged_cells = notebook_runner(notebook_path)

    test_cell = tagged_cells.get("test_ipython_detection")
    assert test_cell is not None, "Could not find test_ipython_detection cell in notebook"

    # Verify the output contains our success message
    output_text = "\n".join(
        str(output.get("text", "")) for output in test_cell.get("outputs", []) if output.get("output_type") == "stream"
    )

    assert "is_interactive_ipython returned: True" in output_text
