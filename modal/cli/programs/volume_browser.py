# Copyright Modal Labs 2025
"""Midnight Commander style TUI for browsing Modal volumes and local filesystem."""

from __future__ import annotations

import asyncio
import multiprocessing
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


class CopyProgressScreen(ModalScreen[None]):
    """A modal screen showing copy progress."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    CopyProgressScreen {
        align: center middle;
    }

    #progress-container {
        width: 70;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #progress-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #progress-status {
        margin-bottom: 1;
    }

    #progress-current-file {
        margin-bottom: 1;
        color: $text-muted;
    }

    #progress-bar {
        margin-bottom: 1;
    }

    #progress-stats {
        color: $text-muted;
        text-align: center;
    }

    #cancel-hint {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }

    #cancel-button {
        margin-top: 1;
        width: 100%;
    }
    """

    def __init__(self, title: str = "Copying Files"):
        super().__init__()
        self.title_text = title
        self.total_files = 0
        self.completed_files = 0
        self.current_file = ""
        self.total_bytes = 0
        self.completed_bytes = 0
        self.cancelled = False

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.title_text, id="progress-title"),
            Label("Preparing...", id="progress-status"),
            Label("", id="progress-current-file"),
            ProgressBar(id="progress-bar", total=100),
            Label("", id="progress-stats"),
            Label("Press ESC to cancel", id="cancel-hint"),
            Button("Cancel", id="cancel-button", variant="error"),
            id="progress-container",
        )

    def action_cancel(self) -> None:
        """Cancel the operation."""
        self.cancelled = True
        self._update_status("Cancelling...")

    @on(Button.Pressed, "#cancel-button")
    def on_cancel_button(self) -> None:
        """Handle cancel button press."""
        self.action_cancel()

    def _update_status(self, status: str) -> None:
        """Update the status message."""
        try:
            self.query_one("#progress-status", Label).update(status)
        except Exception:
            pass  # Screen might be dismissed

    def _update_progress(self) -> None:
        """Update progress display."""
        try:
            # Update current file label
            if self.current_file:
                display_name = self.current_file
                if len(display_name) > 50:
                    display_name = "..." + display_name[-47:]
                self.query_one("#progress-current-file", Label).update(f"  {display_name}")

            # Update progress bar
            if self.total_files > 0:
                progress = (self.completed_files / self.total_files) * 100
                self.query_one("#progress-bar", ProgressBar).update(progress=progress)

            # Update stats
            stats = f"{self.completed_files} / {self.total_files} files"
            if self.total_bytes > 0:
                stats += f" ({humanize_filesize(self.completed_bytes)} / {humanize_filesize(self.total_bytes)})"
            self.query_one("#progress-stats", Label).update(stats)
        except Exception:
            pass  # Screen might be dismissed


class FileViewerScreen(ModalScreen[None]):
    """A modal screen for viewing file contents."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
        Binding("f3", "close", "Close"),
    ]

    CSS = """
    FileViewerScreen {
        align: center middle;
    }

    #viewer-container {
        width: 90%;
        height: 90%;
        border: thick $primary;
        background: $surface;
    }

    #viewer-title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 0 1;
        dock: top;
        height: 1;
    }

    #viewer-content {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }

    #viewer-footer {
        dock: bottom;
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        text-align: center;
    }
    """

    def __init__(self, filename: str, content: str):
        super().__init__()
        self.filename = filename
        self.content = content

    def compose(self) -> ComposeResult:
        from textual.widgets import TextArea

        yield Vertical(
            Label(f" {self.filename} ", id="viewer-title"),
            TextArea(self.content, read_only=True, id="viewer-content", show_line_numbers=True),
            Label("Press ESC, Q, or F3 to close", id="viewer-footer"),
            id="viewer-container",
        )

    def on_mount(self) -> None:
        self.query_one("#viewer-content").focus()

    def action_close(self) -> None:
        self.dismiss(None)


class FilePanel(Static, can_focus=False):
    """A file panel widget showing files in a directory."""

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
        table.add_columns("*", "Name", "Size", "Modified", "Type")
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
            table.add_row(" ", "..", "", "", "dir", key="..")

        # List directory contents
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for entry_path in entries:
                try:
                    item = FileItem.from_local(entry_path)
                    self.items.append(item)
                    mark = "►" if item.path in self.marked_items else " "
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
            table.add_row(" ", "..", "", "", "dir", key="..")

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
                mark = "►" if item.path in self.marked_items else " "
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

    def on_descendant_focus(self, event) -> None:
        """Handle when any child widget (like DataTable) gets focus."""
        self.post_message(self.Focused(self))

    def action_mark(self) -> None:
        """Toggle mark on current item."""
        table = self.query_one("#file-table", DataTable)
        row_index = table.cursor_row

        # Check if items list is populated
        if not self.items:
            self.notify("Directory still loading...", severity="warning")
            return

        if row_index < 0 or row_index >= len(self.items):
            return

        item = self.items[row_index]
        if item.name == "..":
            # Silently skip parent directory
            return

        if item.path in self.marked_items:
            self.marked_items.discard(item.path)
            mark = " "
        else:
            self.marked_items.add(item.path)
            mark = "►"

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
        if current and current.name != "..":
            return [current]
        return []

    def get_selected_item_name(self) -> str:
        """Get a description of selected items for error messages."""
        current = self.get_selected_item()
        if current is None:
            return "nothing"
        if current.name == "..":
            return "parent directory (..)"
        return current.name

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
        Binding("space", "mark", "Mark", priority=True),
        Binding("f3", "view", "View"),
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

    def action_mark(self) -> None:
        """Mark/unmark the current file in the active panel."""
        if self.active_panel:
            self.active_panel.action_mark()
        else:
            self.notify("No active panel", severity="error")

    @work(exclusive=True)
    async def action_view(self) -> None:
        """View the current file."""
        if not self.active_panel:
            return

        item = self.active_panel.get_selected_item()
        if not item:
            self.notify("No file selected", severity="warning")
            return

        if item.name == "..":
            self.notify("Cannot view parent directory", severity="warning")
            return

        if item.is_dir:
            self.notify("Cannot view directory - press Enter to open it", severity="warning")
            return

        # Limit file size for viewing (10 MB max)
        max_size = 10 * 1024 * 1024
        if item.size > max_size:
            self.notify(f"File too large to view ({humanize_filesize(item.size)})", severity="warning")
            return

        try:
            if self.active_panel.panel_type == PanelType.LOCAL:
                # Read local file
                with open(item.path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            else:
                # Read volume file
                if self.volume is None:
                    self.notify("Volume not loaded", severity="error")
                    return

                chunks = []
                async for chunk in self.volume.read_file(item.path):
                    chunks.append(chunk)
                content = b"".join(chunks).decode("utf-8", errors="replace")

            await self.push_screen_wait(FileViewerScreen(item.name, content))
        except Exception as e:
            self.notify(f"Error reading file: {e}", severity="error")

    @work(exclusive=True)
    async def action_copy(self) -> None:
        """Copy files between panels."""
        if not self.active_panel:
            return

        items = self.active_panel.get_marked_items()
        if not items:
            selected = self.active_panel.get_selected_item_name()
            self.notify(
                f"No files selected (cursor on {selected}). Use Space to mark files or navigate to a file.",
                severity="warning",
            )
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
        await self._do_copy_impl(items, source_panel, target_panel)

    async def _do_copy_impl(
        self,
        items: list[FileItem],
        source_panel: FilePanel,
        target_panel: FilePanel,
    ) -> None:
        """Perform the copy operation."""
        # Show progress screen
        progress = CopyProgressScreen("Copying Files...")
        self.app.push_screen(progress)

        try:
            if source_panel.panel_type == PanelType.LOCAL:
                # Local → Volume
                await self._copy_to_volume(items, target_panel.current_path, progress)
            else:
                # Volume → Local
                await self._copy_from_volume(items, target_panel.current_path, progress)

            source_panel.clear_marks()
            target_panel.load_directory()
            self.notify("Copy completed successfully", severity="information")
        except asyncio.CancelledError:
            self.notify("Copy cancelled", severity="warning")
        except Exception as e:
            self.notify(f"Copy failed: {e}", severity="error")
        finally:
            progress.dismiss(None)

    async def _copy_to_volume(self, items: list[FileItem], remote_path: str, progress: CopyProgressScreen) -> None:
        """Copy local files to volume."""
        if self.volume is None:
            raise ValueError("Volume not loaded")

        # First, count all files to copy
        progress._update_status("Scanning files...")
        files_to_copy: list[tuple[Path, str, int]] = []  # (local_path, remote_path, size)

        for item in items:
            if progress.cancelled:
                raise asyncio.CancelledError("Operation cancelled by user")

            local_path = Path(item.path)
            remote_dest = str(PurePosixPath(remote_path) / item.name)

            if item.is_dir:
                for file_path in local_path.rglob("*"):
                    if progress.cancelled:
                        raise asyncio.CancelledError("Operation cancelled by user")
                    if file_path.is_file():
                        rel_path = file_path.relative_to(local_path)
                        file_remote = str(PurePosixPath(remote_dest) / rel_path)
                        files_to_copy.append((file_path, file_remote, file_path.stat().st_size))
            else:
                files_to_copy.append((local_path, remote_dest, item.size))

        if progress.cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

        total_bytes = sum(size for _, _, size in files_to_copy)
        progress.total_files = len(files_to_copy)
        progress.total_bytes = total_bytes
        progress._update_status(f"Copying {len(files_to_copy)} files...")
        progress._update_progress()

        # Now copy files
        async with self.volume.batch_upload(force=True) as batch:
            for local_path, remote_dest, size in files_to_copy:
                if progress.cancelled:
                    raise asyncio.CancelledError("Operation cancelled by user")
                progress.current_file = str(local_path.name)
                progress._update_progress()
                batch.put_file(str(local_path), remote_dest)
                progress.completed_files += 1
                progress.completed_bytes += size
                progress._update_progress()

    async def _copy_from_volume(self, items: list[FileItem], local_path: str, progress: CopyProgressScreen) -> None:
        """Copy volume files to local filesystem with high concurrency.

        Uses a producer-consumer pattern like modal volume get for efficiency,
        with support for cancellation via the progress screen.
        """
        if self.volume is None:
            raise ValueError("Volume not loaded")

        dest_path = Path(local_path)

        # First, enumerate all files to copy (check for cancellation during scan)
        progress._update_status("Scanning files...")
        files_to_copy: list[tuple[str, Path, int]] = []  # (remote_path, local_path, size)

        for item in items:
            if progress.cancelled:
                raise asyncio.CancelledError("Operation cancelled by user")

            if item.is_dir:
                # For directories, list recursively
                async for entry in self.volume.iterdir(item.path, recursive=True):
                    if progress.cancelled:
                        raise asyncio.CancelledError("Operation cancelled by user")
                    if entry.type == FileEntryType.FILE:
                        rel_path = PurePosixPath(entry.path).relative_to(PurePosixPath(item.path).parent)
                        file_dest = dest_path / rel_path
                        files_to_copy.append((entry.path, file_dest, entry.size))
            else:
                file_dest = dest_path / item.name
                files_to_copy.append((item.path, file_dest, item.size))

        if progress.cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

        total_bytes = sum(size for _, _, size in files_to_copy)
        progress.total_files = len(files_to_copy)
        progress.total_bytes = total_bytes
        progress._update_status(f"Downloading {len(files_to_copy)} files...")
        progress._update_progress()

        # Use producer-consumer pattern with queue for better control
        concurrency = min(64, max(16, 2 * multiprocessing.cpu_count()))
        queue: asyncio.Queue[tuple[str, Path, int] | None] = asyncio.Queue()
        download_semaphore = asyncio.Semaphore(concurrency)

        async def producer() -> None:
            """Add files to the download queue."""
            for file_info in files_to_copy:
                if progress.cancelled:
                    break
                await queue.put(file_info)
            # Signal consumers to stop
            for _ in range(concurrency):
                await queue.put(None)

        async def consumer() -> None:
            """Download files from the queue."""
            while True:
                if progress.cancelled:
                    return

                file_info = await queue.get()
                if file_info is None:
                    queue.task_done()
                    return

                remote_path, file_dest, size = file_info
                try:
                    async with download_semaphore:
                        if progress.cancelled:
                            queue.task_done()
                            return

                        file_dest.parent.mkdir(parents=True, exist_ok=True)
                        with open(file_dest, "wb") as f:
                            async for chunk in self.volume.read_file(remote_path):
                                if progress.cancelled:
                                    break
                                f.write(chunk)

                        if not progress.cancelled:
                            progress.completed_files += 1
                            progress.completed_bytes += size
                            progress.current_file = PurePosixPath(remote_path).name
                            progress._update_progress()
                finally:
                    queue.task_done()

        # Run producer and consumers concurrently
        consumers = [asyncio.create_task(consumer()) for _ in range(concurrency)]
        producer_task = asyncio.create_task(producer())

        try:
            await producer_task
            await queue.join()
        finally:
            # Cancel any remaining consumer tasks
            for task in consumers:
                task.cancel()
            # Wait for consumers to finish
            await asyncio.gather(*consumers, return_exceptions=True)

        if progress.cancelled:
            raise asyncio.CancelledError("Operation cancelled by user")

    @work(exclusive=True)
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

    @work(exclusive=True)
    async def action_delete(self) -> None:
        """Delete selected files."""
        if not self.active_panel:
            return

        items = self.active_panel.get_marked_items()
        if not items:
            selected = self.active_panel.get_selected_item_name()
            self.notify(
                f"No files selected (cursor on {selected}). Use Space to mark files or navigate to a file.",
                severity="warning",
            )
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

        await self._do_delete_impl(items)

    async def _do_delete_impl(self, items: list[FileItem]) -> None:
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

    @work(exclusive=True)
    async def action_move(self) -> None:
        """Move files (copy + delete source)."""
        if not self.active_panel:
            return

        items = self.active_panel.get_marked_items()
        if not items:
            selected = self.active_panel.get_selected_item_name()
            self.notify(
                f"No files selected (cursor on {selected}). Use Space to mark files or navigate to a file.",
                severity="warning",
            )
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

        await self._do_move_impl(items, source_panel, target_panel)

    async def _do_move_impl(
        self,
        items: list[FileItem],
        source_panel: FilePanel,
        target_panel: FilePanel,
    ) -> None:
        """Perform the move operation (copy + delete)."""
        # Show progress screen
        progress = CopyProgressScreen("Moving Files...")
        self.app.push_screen(progress)

        try:
            # First copy
            if source_panel.panel_type == PanelType.LOCAL:
                await self._copy_to_volume(items, target_panel.current_path, progress)
            else:
                await self._copy_from_volume(items, target_panel.current_path, progress)

            # Then delete source
            progress._update_status("Deleting source files...")
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
        except asyncio.CancelledError:
            self.notify("Move cancelled (files may have been partially copied)", severity="warning")
        except Exception as e:
            self.notify(f"Move failed: {e}", severity="error")
        finally:
            progress.dismiss(None)


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
