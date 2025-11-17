# Copyright Modal Labs 2025
import pytest


def test_is_interactive_ipython_in_real_notebook(test_dir):
    import jupytext
    from nbclient.exceptions import CellExecutionError

    """Integration test: Run actual notebook to verify is_interactive_ipython returns True."""
    import nbformat
    from nbclient.client import NotebookClient

    notebook_path = test_dir / "supports" / "notebooks" / "ipython_detection.notebook.py"
    output_notebook_path = notebook_path.with_suffix(".output.ipynb")

    # Read and execute the notebook
    nb = jupytext.read(notebook_path)
    client = NotebookClient(nb)

    try:
        client.execute()
    except CellExecutionError:  # type: ignore  # noqa: F821
        # Write the failed notebook for debugging
        nbformat.write(nb, output_notebook_path)
        pytest.fail(
            f"""Notebook execution failed. The test assertion failed, meaning
is_interactive_ipython() did not return True in the notebook environment.

Inspect the output notebook: {output_notebook_path}
"""
        )

    # Find the test cell
    test_cell = None
    for cell in nb["cells"]:
        if "test_ipython_detection" in cell["metadata"].get("tags", []):
            test_cell = cell
            break

    assert test_cell is not None, "Could not find test cell in notebook"

    # Check that the cell executed successfully
    # The notebook has an assert statement that will fail if is_interactive_ipython() returns False
    # If we got here, the assertion passed, meaning is_interactive_ipython() returned True

    # Verify the output contains our success message
    output_text = "\n".join(
        str(output.get("text", "")) for output in test_cell.get("outputs", []) if output.get("output_type") == "stream"
    )

    assert "is_interactive_ipython returned: True" in output_text
