# Copyright Modal Labs 2026
from pathlib import Path


def write_modules(output_dir: Path, modules: dict[str, str]):
    # Write to files
    created_dirs = set()

    for module_name, code in modules.items():
        # Convert module name to file path
        module_path = output_dir / (module_name.replace(".", "/") + ".py")

        # Create parent directories and track them
        module_path.parent.mkdir(parents=True, exist_ok=True)

        # Track all directories in the path for __init__.py creation
        current = module_path.parent
        while current != output_dir and current not in created_dirs:
            created_dirs.add(current)
            current = current.parent

        # Write file
        module_path.write_text(code)
        yield module_path

    # Create __init__.py files in all package directories
    for dir_path in created_dirs:
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Auto-generated package file\n")
