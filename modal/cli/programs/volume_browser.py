# Copyright Modal Labs 2025
"""Midnight Commander style TUI for browsing Modal volumes and local filesystem."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, ProgressBar, Static

from modal.environments import ensure_env
from modal.volume import FileEntry, FileEntryType, _Volume


class PanelType(Enum):
    LOCAL = "local"
    VOLUME = "volume"


@dataclass
class FileItem:
    """Represents a file or directory item in either panel."""

    name: str
    path: str
    is_dir: bool
    size: int
    mtime: datetime
    type: str  # "file", "dir", "link", etc.

    @classmethod
    def from_local(cls, path: Path) -> "FileItem":
        """Create FileItem from local filesystem path."""
        stat_info = path.stat()
        is_dir = path.is_dir()
        file_type = "dir" if is_dir else "file"
        if path.is_symlink():
            file_type = "link"
        return cls(
            name=path.name,
            path=str(path),
            is_dir=is_dir,
            size=stat_info.st_size if not is_dir else 0,
            mtime=datetime.fromtimestamp(stat_info.st_mtime),
            type=file_type,
        )

    @classmethod
    def from_volume_entry(cls, entry: FileEntry) -> "FileItem":
        """Create FileItem from Modal Volume FileEntry."""
        is_dir = entry.type == FileEntryType.DIRECTORY
        type_map = {
            FileEntryType.FILE: "file",
            FileEntryType.DIRECTORY: "dir",
            FileEntryType.SYMLINK: "link",
            FileEntryType.FIFO: "fifo",
            FileEntryType.SOCKET: "socket",
        }
        return cls(
            name=PurePosixPath(entry.path).name or entry.path,
            path=entry.path,
            is_dir=is_dir,
            size=entry.size,
            mtime=datetime.fromtimestamp(entry.mtime),
            type=type_map.get(entry.type, "file"),
        )

    @classmethod
    def parent_dir(cls, current_path: str) -> "FileItem":
        """Create a parent directory entry (..)."""
        return cls(
            name="..",
            path=str(Path(current_path).parent) if current_path != "/" else "/",
            is_dir=True,
            size=0,
            mtime=datetime.now(),
            type="dir",
        )


def humanize_filesize(value: int) -> str:
    """Convert bytes to human readable string."""
    if value < 0:
        return ""
    suffix = (" KiB", " MiB", " GiB", " TiB", " PiB")
    format_str = "%.1f"
    base = 1024
    bytes_ = float(value)
    if bytes_ < base:
        return f"{bytes_:0.0f} B"
    for i, s in enumerate(suffix):
        unit = base ** (i + 2)
        if bytes_ < unit:
            return format_str % (base * bytes_ / unit) + s
    return format_str % (base * bytes_ / (base ** len(suffix))) + suffix[-1]


class ConfirmDialog(ModalScreen[bool]):
    """A modal dialog for confirming actions."""

    def __init__(self, message: str, title: str = "Confirm"):
        super().__init__()
        self.message = message
        self.title_text = title

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.title_text, id="dialog-title"),
            Label(self.message, id="dialog-message"),
            Horizontal(
                Button("Yes", variant="primary", id="yes"),
                Button("No", variant="default", id="no"),
                id="dialog-buttons",
            ),
            id="dialog",
        )

    @on(Button.Pressed, "#yes")
    def confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def cancel(self) -> None:
        self.dismiss(False)


class InputDialog(ModalScreen[Optional[str]]):
    """A modal dialog for text input."""

    def __init__(self, message: str, title: str = "Input", default: str = ""):
        super().__init__()
        self.message = message
        self.title_text = title
        self.default = default

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.title_text, id="dialog-title"),
            Label(self.message, id="dialog-message"),
            Input(value=self.default, id="input-field"),
            Horizontal(
                Button("OK", variant="primary", id="ok"),
                Button("Cancel", variant="default", id="cancel"),
                id="dialog-buttons",
            ),
            id="dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#input-field", Input).focus()

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def confirm(self) -> None:
        value = self.query_one("#input-field", Input).value
        self.dismiss(value)

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class ProgressDialog(ModalScreen[None]):
    """A modal dialog showing progress."""

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Operation in Progress", id="dialog-title"),
            Label(self.message, id="progress-message"),
            ProgressBar(id="progress-bar"),
            id="dialog",
        )

    def update_progress(self, message: str, progress: float = -1) -> None:
        self.query_one("#progress-message", Label).update(message)
        if progress >= 0:
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.update(progress=progress)


class FilePanel(Static):
    """A file panel widget showing files in a directory."""

    BINDINGS = [
        Binding("enter", "select", "Open/Select"),
        Binding("space", "mark", "Mark"),
    ]

    current_path: reactive[str] = reactive(".")
    is_active: reactive[bool] = reactive(False)

    class Selected(Message):
        """Message sent when an item is selected."""

        def __init__(self, panel: "FilePanel", item: FileItem) -> None:
            super().__init__()
            self.panel = panel
            self.item = item

    class PathChanged(Message):
        """Message sent when the current path changes."""

        def __init__(self, panel: "FilePanel", path: str) -> None:
            super().__init__()
            self.panel = panel
            self.path = path

    class Focused(Message):
        """Message sent when panel gains focus."""

        def __init__(self, panel: "FilePanel") -> None:
            super().__init__()
            self.panel = panel

    def __init__(
        self,
        panel_type: PanelType,
        title: str = "",
        volume: Optional[_Volume] = None,
        initial_path: str = ".",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.panel_type = panel_type
        self.title_text = title
        self.volume = volume
        self.current_path = initial_path
        self.items: list[FileItem] = []
        self.marked_items: set[str] = set()
        self._loading = False

    def compose(self) -> ComposeResult:
        yield Label(self.title_text, id="panel-title")
        yield Label(self.current_path, id="panel-path")
        yield DataTable(id="file-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#file-table", DataTable)
        table.add_columns("", "Name", "Size", "Modified", "Type")
        table.cursor_type = "row"
        self.load_directory()

    def watch_is_active(self, is_active: bool) -> None:
        """Update styling when active state changes."""
        self.set_class(is_active, "active-panel")

    @work(exclusive=True)
    async def load_directory(self) -> None:
        """Load directory contents."""
        self._loading = True
        table = self.query_one("#file-table", DataTable)
        table.clear()
        self.items = []

        try:
            if self.panel_type == PanelType.LOCAL:
                await self._load_local_directory()
            else:
                await self._load_volume_directory()
        except Exception as e:
            self.notify(f"Error loading directory: {e}", severity="error")
        finally:
            self._loading = False

    async def _load_local_directory(self) -> None:
        """Load local filesystem directory."""
        table = self.query_one("#file-table", DataTable)
        path = Path(self.current_path).resolve()
        self.current_path = str(path)

        # Update path label
        self.query_one("#panel-path", Label).update(str(path))

        # Add parent directory entry
        if path != path.parent:
            parent_item = FileItem.parent_dir(str(path))
            self.items.append(parent_item)
            table.add_row("", "..", "", "", "dir", key="..")

        # List directory contents
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for entry_path in entries:
                try:
                    item = FileItem.from_local(entry_path)
                    self.items.append(item)
                    mark = "●" if item.path in self.marked_items else ""
                    size = humanize_filesize(item.size) if not item.is_dir else "<DIR>"
                    mtime = item.mtime.strftime("%Y-%m-%d %H:%M")
                    table.add_row(mark, item.name, size, mtime, item.type, key=item.name)
                except (PermissionError, OSError):
                    continue
        except PermissionError:
            self.notify("Permission denied", severity="error")

    async def _load_volume_directory(self) -> None:
        """Load Modal Volume directory."""
        table = self.query_one("#file-table", DataTable)

        # Update path label
        self.query_one("#panel-path", Label).update(self.current_path)

        # Add parent directory entry if not at root
        if self.current_path != "/":
            parent_path = str(PurePosixPath(self.current_path).parent)
            if not parent_path or parent_path == ".":
                parent_path = "/"
            parent_item = FileItem(
                name="..",
                path=parent_path,
                is_dir=True,
                size=0,
                mtime=datetime.now(),
                type="dir",
            )
            self.items.append(parent_item)
            table.add_row("", "..", "", "", "dir", key="..")

        if self.volume is None:
            return

        try:
            # List volume contents (non-recursive to get immediate children)
            entries = await self.volume.listdir(self.current_path, recursive=False)

            # Sort: directories first, then by name
            entries.sort(key=lambda e: (e.type != FileEntryType.DIRECTORY, e.path.lower()))

            for entry in entries:
                item = FileItem.from_volume_entry(entry)
                self.items.append(item)
                mark = "●" if item.path in self.marked_items else ""
                size = humanize_filesize(item.size) if not item.is_dir else "<DIR>"
                mtime = item.mtime.strftime("%Y-%m-%d %H:%M")
                table.add_row(mark, item.name, size, mtime, item.type, key=item.name)
        except Exception as e:
            self.notify(f"Error listing volume: {e}", severity="error")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection (Enter key)."""
        if event.row_key is None or not self.items:
            return

        # Find the item by matching row index
        row_index = event.cursor_row
        if 0 <= row_index < len(self.items):
            item = self.items[row_index]
            if item.is_dir:
                self.current_path = item.path if self.panel_type == PanelType.VOLUME else str(Path(item.path).resolve())
                self.post_message(self.PathChanged(self, self.current_path))
                self.load_directory()
            else:
                self.post_message(self.Selected(self, item))

    def on_focus(self) -> None:
        """Handle focus event."""
        self.post_message(self.Focused(self))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Forward focus to parent."""
        self.post_message(self.Focused(self))

    def action_mark(self) -> None:
        """Toggle mark on current item."""
        table = self.query_one("#file-table", DataTable)
        row_index = table.cursor_row
        if 0 <= row_index < len(self.items):
            item = self.items[row_index]
            if item.name == "..":
                return

            if item.path in self.marked_items:
                self.marked_items.discard(item.path)
                mark = ""
            else:
                self.marked_items.add(item.path)
                mark = "●"

            # Update the mark column
            table.update_cell_at(Coordinate(row_index, 0), mark)

            # Move cursor down
            if row_index < len(self.items) - 1:
                table.move_cursor(row=row_index + 1)

    def get_selected_item(self) -> Optional[FileItem]:
        """Get the currently highlighted item."""
        table = self.query_one("#file-table", DataTable)
        row_index = table.cursor_row
        if 0 <= row_index < len(self.items):
            return self.items[row_index]
        return None

    def get_marked_items(self) -> list[FileItem]:
        """Get all marked items, or current item if none marked."""
        if self.marked_items:
            return [item for item in self.items if item.path in self.marked_items]
        current = self.get_selected_item()
        return [current] if current and current.name != ".." else []

    def clear_marks(self) -> None:
        """Clear all marks."""
        self.marked_items.clear()
        self.load_directory()


class VolumeBrowserApp(App):
    """Midnight Commander style file browser for Modal Volumes."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 1;
        grid-gutter: 1;
    }

    FilePanel {
        border: solid green;
        height: 100%;
    }

    FilePanel.active-panel {
        border: double cyan;
    }

    #panel-title {
        text-align: center;
        text-style: bold;
        background: $surface;
        color: $text;
        padding: 0 1;
    }

    #panel-path {
        text-align: left;
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
    }

    #file-table {
        height: 1fr;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text;
    }

    #dialog {
        align: center middle;
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #dialog-message {
        margin-bottom: 1;
    }

    #dialog-buttons {
        align: center middle;
        height: auto;
        margin-top: 1;
    }

    #dialog-buttons Button {
        margin: 0 1;
    }

    ConfirmDialog, InputDialog, ProgressDialog {
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("tab", "switch_panel", "Switch Panel"),
        Binding("f5", "copy", "Copy"),
        Binding("f6", "move", "Move"),
        Binding("f7", "mkdir", "MkDir"),
        Binding("f8", "delete", "Delete"),
        Binding("f10", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        volume_name: str,
        environment_name: Optional[str] = None,
        local_path: str = ".",
        volume_path: str = "/",
    ):
        super().__init__()
        self.volume_name = volume_name
        self.environment_name = environment_name
        self.local_path = os.path.abspath(local_path)
        self.volume_path = volume_path
        self.volume: Optional[_Volume] = None
        self.active_panel: Optional[FilePanel] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield FilePanel(
            PanelType.LOCAL,
            title=f"Local: {self.local_path}",
            initial_path=self.local_path,
            id="local-panel",
        )
        yield FilePanel(
            PanelType.VOLUME,
            title=f"Volume: {self.volume_name}",
            initial_path=self.volume_path,
            id="volume-panel",
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app and load the volume."""
        self.title = f"Modal Volume Browser - {self.volume_name}"

        # Load volume
        try:
            self.volume = _Volume.from_name(self.volume_name, environment_name=self.environment_name)
            await self.volume.hydrate()

            # Set volume on the volume panel
            volume_panel = self.query_one("#volume-panel", FilePanel)
            volume_panel.volume = self.volume
            volume_panel.load_directory()
        except Exception as e:
            self.notify(f"Error loading volume: {e}", severity="error")

        # Focus local panel by default
        local_panel = self.query_one("#local-panel", FilePanel)
        self.active_panel = local_panel
        local_panel.is_active = True
        local_panel.query_one("#file-table", DataTable).focus()

    def on_file_panel_focused(self, event: FilePanel.Focused) -> None:
        """Handle panel focus changes."""
        if self.active_panel:
            self.active_panel.is_active = False
        self.active_panel = event.panel
        event.panel.is_active = True

    def action_switch_panel(self) -> None:
        """Switch between panels."""
        local_panel = self.query_one("#local-panel", FilePanel)
        volume_panel = self.query_one("#volume-panel", FilePanel)

        if self.active_panel == local_panel:
            self.active_panel.is_active = False
            self.active_panel = volume_panel
            volume_panel.is_active = True
            volume_panel.query_one("#file-table", DataTable).focus()
        else:
            self.active_panel.is_active = False
            self.active_panel = local_panel
            local_panel.is_active = True
            local_panel.query_one("#file-table", DataTable).focus()

    def action_refresh(self) -> None:
        """Refresh both panels."""
        self.query_one("#local-panel", FilePanel).load_directory()
        self.query_one("#volume-panel", FilePanel).load_directory()

    async def action_copy(self) -> None:
        """Copy files between panels."""
        if not self.active_panel:
            return

        items = self.active_panel.get_marked_items()
        if not items:
            self.notify("No files selected", severity="warning")
            return

        source_panel = self.active_panel
        target_panel = (
            self.query_one("#volume-panel", FilePanel)
            if source_panel.panel_type == PanelType.LOCAL
            else self.query_one("#local-panel", FilePanel)
        )

        # Get confirmation
        file_list = ", ".join(item.name for item in items[:3])
        if len(items) > 3:
            file_list += f" and {len(items) - 3} more"

        direction = "local → volume" if source_panel.panel_type == PanelType.LOCAL else "volume → local"

        confirmed = await self.push_screen_wait(
            ConfirmDialog(
                f"Copy {file_list} ({direction})?",
                title="Confirm Copy",
            )
        )

        if not confirmed:
            return

        # Perform copy
        self._do_copy(items, source_panel, target_panel)

    @work(exclusive=True)
    async def _do_copy(
        self,
        items: list[FileItem],
        source_panel: FilePanel,
        target_panel: FilePanel,
    ) -> None:
        """Perform the copy operation."""
        try:
            if source_panel.panel_type == PanelType.LOCAL:
                # Local → Volume
                await self._copy_to_volume(items, target_panel.current_path)
            else:
                # Volume → Local
                await self._copy_from_volume(items, target_panel.current_path)

            source_panel.clear_marks()
            target_panel.load_directory()
            self.notify("Copy completed successfully", severity="information")
        except Exception as e:
            self.notify(f"Copy failed: {e}", severity="error")

    async def _copy_to_volume(self, items: list[FileItem], remote_path: str) -> None:
        """Copy local files to volume."""
        if self.volume is None:
            raise ValueError("Volume not loaded")

        async with self.volume.batch_upload(force=True) as batch:
            for item in items:
                local_path = Path(item.path)
                remote_dest = str(PurePosixPath(remote_path) / item.name)

                if item.is_dir:
                    batch.put_directory(str(local_path), remote_dest)
                else:
                    batch.put_file(str(local_path), remote_dest)

    async def _copy_from_volume(self, items: list[FileItem], local_path: str) -> None:
        """Copy volume files to local filesystem."""
        if self.volume is None:
            raise ValueError("Volume not loaded")

        dest_path = Path(local_path)

        for item in items:
            if item.is_dir:
                # For directories, download recursively
                async for entry in self.volume.iterdir(item.path, recursive=True):
                    if entry.type == FileEntryType.FILE:
                        rel_path = PurePosixPath(entry.path).relative_to(PurePosixPath(item.path).parent)
                        file_dest = dest_path / rel_path
                        file_dest.parent.mkdir(parents=True, exist_ok=True)

                        with open(file_dest, "wb") as f:
                            async for chunk in self.volume.read_file(entry.path):
                                f.write(chunk)
            else:
                file_dest = dest_path / item.name
                file_dest.parent.mkdir(parents=True, exist_ok=True)

                with open(file_dest, "wb") as f:
                    async for chunk in self.volume.read_file(item.path):
                        f.write(chunk)

    async def action_mkdir(self) -> None:
        """Create a new directory."""
        if not self.active_panel:
            return

        if self.active_panel.panel_type == PanelType.VOLUME:
            self.notify("Creating directories in volumes is not supported", severity="warning")
            return

        name = await self.push_screen_wait(InputDialog("Enter directory name:", title="Create Directory"))

        if not name:
            return

        try:
            new_path = Path(self.active_panel.current_path) / name
            new_path.mkdir(parents=True, exist_ok=True)
            self.active_panel.load_directory()
            self.notify(f"Created directory: {name}", severity="information")
        except Exception as e:
            self.notify(f"Failed to create directory: {e}", severity="error")

    async def action_delete(self) -> None:
        """Delete selected files."""
        if not self.active_panel:
            return

        items = self.active_panel.get_marked_items()
        if not items:
            self.notify("No files selected", severity="warning")
            return

        file_list = ", ".join(item.name for item in items[:3])
        if len(items) > 3:
            file_list += f" and {len(items) - 3} more"

        confirmed = await self.push_screen_wait(
            ConfirmDialog(
                f"Delete {file_list}? This cannot be undone!",
                title="Confirm Delete",
            )
        )

        if not confirmed:
            return

        self._do_delete(items)

    @work(exclusive=True)
    async def _do_delete(self, items: list[FileItem]) -> None:
        """Perform the delete operation."""
        if not self.active_panel:
            return

        try:
            if self.active_panel.panel_type == PanelType.LOCAL:
                import shutil

                for item in items:
                    path = Path(item.path)
                    if item.is_dir:
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            else:
                if self.volume is None:
                    raise ValueError("Volume not loaded")

                for item in items:
                    await self.volume.remove_file(item.path, recursive=item.is_dir)

            self.active_panel.clear_marks()
            self.active_panel.load_directory()
            self.notify("Delete completed successfully", severity="information")
        except Exception as e:
            self.notify(f"Delete failed: {e}", severity="error")

    async def action_move(self) -> None:
        """Move files (copy + delete source)."""
        if not self.active_panel:
            return

        items = self.active_panel.get_marked_items()
        if not items:
            self.notify("No files selected", severity="warning")
            return

        source_panel = self.active_panel
        target_panel = (
            self.query_one("#volume-panel", FilePanel)
            if source_panel.panel_type == PanelType.LOCAL
            else self.query_one("#local-panel", FilePanel)
        )

        file_list = ", ".join(item.name for item in items[:3])
        if len(items) > 3:
            file_list += f" and {len(items) - 3} more"

        direction = "local → volume" if source_panel.panel_type == PanelType.LOCAL else "volume → local"

        confirmed = await self.push_screen_wait(
            ConfirmDialog(
                f"Move {file_list} ({direction})? Source will be deleted!",
                title="Confirm Move",
            )
        )

        if not confirmed:
            return

        self._do_move(items, source_panel, target_panel)

    @work(exclusive=True)
    async def _do_move(
        self,
        items: list[FileItem],
        source_panel: FilePanel,
        target_panel: FilePanel,
    ) -> None:
        """Perform the move operation (copy + delete)."""
        try:
            # First copy
            if source_panel.panel_type == PanelType.LOCAL:
                await self._copy_to_volume(items, target_panel.current_path)
            else:
                await self._copy_from_volume(items, target_panel.current_path)

            # Then delete source
            if source_panel.panel_type == PanelType.LOCAL:
                import shutil

                for item in items:
                    path = Path(item.path)
                    if item.is_dir:
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            else:
                if self.volume is None:
                    raise ValueError("Volume not loaded")

                for item in items:
                    await self.volume.remove_file(item.path, recursive=item.is_dir)

            source_panel.clear_marks()
            source_panel.load_directory()
            target_panel.load_directory()
            self.notify("Move completed successfully", severity="information")
        except Exception as e:
            self.notify(f"Move failed: {e}", severity="error")


def run_volume_browser(
    volume_name: str,
    environment_name: Optional[str] = None,
    local_path: str = ".",
    volume_path: str = "/",
) -> None:
    """Run the volume browser TUI."""
    ensure_env(environment_name)
    app = VolumeBrowserApp(
        volume_name=volume_name,
        environment_name=environment_name,
        local_path=local_path,
        volume_path=volume_path,
    )
    app.run()
