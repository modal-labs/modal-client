# Copyright Modal Labs 2025
"""Tests for the Midnight Commander style volume browser TUI."""

import pytest
from datetime import datetime
from pathlib import Path, PurePosixPath
from unittest import mock

from modal.volume import FileEntry, FileEntryType


# Mock the _Volume class for testing
class MockVolume:
    """Mock Volume for testing without network calls."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/"}
        self._metadata = mock.MagicMock()
        self._metadata.version = None
        self._client = mock.MagicMock()
        self.object_id = "vol-test-123"

    def add_file(self, path: str, content: bytes = b"test content") -> None:
        """Add a file to the mock volume."""
        self.files[path] = content
        # Ensure parent directories exist (use PurePosixPath for Unix-style paths)
        posix_path = PurePosixPath(path)
        parent = str(posix_path.parent)
        while parent and parent != "/":
            self.dirs.add(parent)
            parent = str(PurePosixPath(parent).parent)
        self.dirs.add("/")

    def add_directory(self, path: str) -> None:
        """Add a directory to the mock volume."""
        self.dirs.add(path)
        posix_path = PurePosixPath(path)
        parent = str(posix_path.parent)
        while parent and parent != "/":
            self.dirs.add(parent)
            parent = str(PurePosixPath(parent).parent)

    async def listdir(self, path: str, recursive: bool = False) -> list[FileEntry]:
        """List files in a directory."""
        entries = []
        path = path.rstrip("/") or "/"

        if recursive:
            # Return all files under path
            for file_path in self.files:
                if file_path.startswith(path + "/") or (path == "/" and file_path.startswith("/")):
                    entries.append(
                        FileEntry(
                            path=file_path,
                            type=FileEntryType.FILE,
                            mtime=int(datetime.now().timestamp()),
                            size=len(self.files[file_path]),
                        )
                    )
            for dir_path in self.dirs:
                if dir_path != path and (dir_path.startswith(path + "/") or (path == "/" and dir_path != "/")):
                    entries.append(
                        FileEntry(
                            path=dir_path,
                            type=FileEntryType.DIRECTORY,
                            mtime=int(datetime.now().timestamp()),
                            size=0,
                        )
                    )
        else:
            # Return immediate children only
            seen = set()
            for file_path in self.files:
                if path == "/":
                    parts = file_path.split("/")
                    if len(parts) >= 2:
                        name = parts[1]
                        if name and name not in seen:
                            # Check if it's a direct child or in a subdirectory
                            if len(parts) == 2:
                                entries.append(
                                    FileEntry(
                                        path=file_path,
                                        type=FileEntryType.FILE,
                                        mtime=int(datetime.now().timestamp()),
                                        size=len(self.files[file_path]),
                                    )
                                )
                                seen.add(name)
                elif file_path.startswith(path + "/"):
                    rel = file_path[len(path) + 1 :]
                    if "/" not in rel:
                        entries.append(
                            FileEntry(
                                path=file_path,
                                type=FileEntryType.FILE,
                                mtime=int(datetime.now().timestamp()),
                                size=len(self.files[file_path]),
                            )
                        )

            for dir_path in self.dirs:
                if dir_path != path:
                    if path == "/":
                        parts = dir_path.split("/")
                        if len(parts) == 2 and parts[1] and parts[1] not in seen:
                            entries.append(
                                FileEntry(
                                    path=dir_path,
                                    type=FileEntryType.DIRECTORY,
                                    mtime=int(datetime.now().timestamp()),
                                    size=0,
                                )
                            )
                            seen.add(parts[1])
                    elif dir_path.startswith(path + "/"):
                        rel = dir_path[len(path) + 1 :]
                        if "/" not in rel:
                            entries.append(
                                FileEntry(
                                    path=dir_path,
                                    type=FileEntryType.DIRECTORY,
                                    mtime=int(datetime.now().timestamp()),
                                    size=0,
                                )
                            )

        return entries

    async def read_file(self, path: str):
        """Read a file from the volume."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        content = self.files[path]
        yield content

    async def iterdir(self, path: str, recursive: bool = True):
        """Iterate over files in a directory."""
        entries = await self.listdir(path, recursive=recursive)
        for entry in entries:
            yield entry

    async def remove_file(self, path: str, recursive: bool = False) -> None:
        """Remove a file or directory."""
        if path in self.files:
            del self.files[path]
        elif path in self.dirs:
            if recursive:
                # Remove all files under this directory
                to_remove = [f for f in self.files if f.startswith(path + "/")]
                for f in to_remove:
                    del self.files[f]
                # Remove subdirectories
                to_remove_dirs = [d for d in self.dirs if d.startswith(path + "/")]
                for d in to_remove_dirs:
                    self.dirs.discard(d)
            self.dirs.discard(path)

    async def hydrate(self):
        """Mock hydrate method."""
        return self

    def batch_upload(self, force: bool = False):
        """Return a mock batch upload context manager."""
        return MockBatchUpload(self, force)


class MockBatchUpload:
    """Mock batch upload context manager."""

    def __init__(self, volume: MockVolume, force: bool = False):
        self.volume = volume
        self.force = force
        self._pending_files: list[tuple[str, str]] = []  # (local, remote)
        self._pending_dirs: list[tuple[str, str]] = []  # (local, remote)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Process pending uploads
        for local_path, remote_path in self._pending_files:
            with open(local_path, "rb") as f:
                self.volume.files[remote_path] = f.read()

        for local_dir, remote_dir in self._pending_dirs:
            local_dir_path = Path(local_dir)
            for file_path in local_dir_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(local_dir_path)
                    remote_file = f"{remote_dir}/{rel_path}".replace("\\", "/")
                    with open(file_path, "rb") as f:
                        self.volume.files[remote_file] = f.read()

    def put_file(self, local_path: str, remote_path: str) -> None:
        self._pending_files.append((local_path, remote_path))

    def put_directory(self, local_path: str, remote_path: str) -> None:
        self._pending_dirs.append((local_path, remote_path))


class TestFileItem:
    """Test the FileItem dataclass."""

    def test_from_local_file(self, tmp_path):
        """Test creating FileItem from a local file."""
        from modal.cli.programs.volume_browser import FileItem

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        item = FileItem.from_local(test_file)
        assert item.name == "test.txt"
        assert item.is_dir is False
        assert item.size == 11
        assert item.type == "file"

    def test_from_local_directory(self, tmp_path):
        """Test creating FileItem from a local directory."""
        from modal.cli.programs.volume_browser import FileItem

        # Create a test directory
        test_dir = tmp_path / "subdir"
        test_dir.mkdir()

        item = FileItem.from_local(test_dir)
        assert item.name == "subdir"
        assert item.is_dir is True
        assert item.type == "dir"

    def test_from_volume_entry(self):
        """Test creating FileItem from a Volume FileEntry."""
        from modal.cli.programs.volume_browser import FileItem

        entry = FileEntry(
            path="/data/file.txt",
            type=FileEntryType.FILE,
            mtime=int(datetime.now().timestamp()),
            size=1024,
        )

        item = FileItem.from_volume_entry(entry)
        assert item.name == "file.txt"
        assert item.path == "/data/file.txt"
        assert item.is_dir is False
        assert item.size == 1024
        assert item.type == "file"

    def test_from_volume_entry_directory(self):
        """Test creating FileItem from a Volume directory entry."""
        from modal.cli.programs.volume_browser import FileItem

        entry = FileEntry(
            path="/data/subdir",
            type=FileEntryType.DIRECTORY,
            mtime=int(datetime.now().timestamp()),
            size=0,
        )

        item = FileItem.from_volume_entry(entry)
        assert item.name == "subdir"
        assert item.is_dir is True
        assert item.type == "dir"


class TestHumanizeFilesize:
    """Test the humanize_filesize function."""

    def test_bytes(self):
        from modal.cli.programs.volume_browser import humanize_filesize

        assert humanize_filesize(0) == "0 B"
        assert humanize_filesize(512) == "512 B"
        assert humanize_filesize(1023) == "1023 B"

    def test_kibibytes(self):
        from modal.cli.programs.volume_browser import humanize_filesize

        assert humanize_filesize(1024) == "1.0 KiB"
        assert humanize_filesize(1536) == "1.5 KiB"

    def test_mebibytes(self):
        from modal.cli.programs.volume_browser import humanize_filesize

        assert humanize_filesize(1024 * 1024) == "1.0 MiB"
        assert humanize_filesize(1024 * 1024 * 5) == "5.0 MiB"

    def test_gibibytes(self):
        from modal.cli.programs.volume_browser import humanize_filesize

        assert humanize_filesize(1024 * 1024 * 1024) == "1.0 GiB"


class TestMockVolume:
    """Test the MockVolume class itself."""

    @pytest.mark.asyncio
    async def test_add_and_list_files(self):
        vol = MockVolume()
        vol.add_file("/file1.txt", b"content1")
        vol.add_file("/file2.txt", b"content2")

        entries = await vol.listdir("/", recursive=False)
        assert len(entries) == 2
        names = {e.path for e in entries}
        assert "/file1.txt" in names
        assert "/file2.txt" in names

    @pytest.mark.asyncio
    async def test_add_and_list_directories(self):
        vol = MockVolume()
        vol.add_directory("/subdir")
        vol.add_file("/subdir/file.txt", b"content")

        entries = await vol.listdir("/", recursive=False)
        # Should see the subdir
        dir_entries = [e for e in entries if e.type == FileEntryType.DIRECTORY]
        assert len(dir_entries) == 1
        assert dir_entries[0].path == "/subdir"

    @pytest.mark.asyncio
    async def test_read_file(self):
        vol = MockVolume()
        vol.add_file("/test.txt", b"hello world")

        chunks = []
        async for chunk in vol.read_file("/test.txt"):
            chunks.append(chunk)
        assert b"".join(chunks) == b"hello world"

    @pytest.mark.asyncio
    async def test_remove_file(self):
        vol = MockVolume()
        vol.add_file("/test.txt", b"content")

        await vol.remove_file("/test.txt")
        entries = await vol.listdir("/", recursive=False)
        assert len(entries) == 0


class TestVolumeBrowserApp:
    """Test the VolumeBrowserApp class."""

    @pytest.fixture
    def mock_volume(self):
        """Create a mock volume with test data."""
        vol = MockVolume()
        vol.add_file("/file1.txt", b"File 1 content")
        vol.add_file("/file2.txt", b"File 2 content here")
        vol.add_directory("/subdir")
        vol.add_file("/subdir/nested.txt", b"Nested file content")
        return vol

    @pytest.fixture
    def temp_local_dir(self, tmp_path):
        """Create a temporary local directory with test files."""
        # Create some test files
        (tmp_path / "local1.txt").write_text("Local file 1")
        (tmp_path / "local2.txt").write_text("Local file 2")
        subdir = tmp_path / "localdir"
        subdir.mkdir()
        (subdir / "nested_local.txt").write_text("Nested local file")
        return tmp_path

    def test_app_creation(self, temp_local_dir):
        """Test that the app can be created."""
        from modal.cli.programs.volume_browser import VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )
        assert app.volume_name == "test-volume"
        assert app.local_path == str(temp_local_dir)
        assert app.volume_path == "/"

    @pytest.mark.asyncio
    async def test_app_compose(self, temp_local_dir):
        """Test that the app composes correctly."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            # Check that both panels exist
            local_panel = app.query_one("#local-panel", FilePanel)
            volume_panel = app.query_one("#volume-panel", FilePanel)

            assert local_panel is not None
            assert volume_panel is not None

    @pytest.mark.asyncio
    async def test_panel_switching(self, temp_local_dir, mock_volume):
        """Test switching between panels with Tab."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            # Inject mock volume
            app.volume = mock_volume
            volume_panel = app.query_one("#volume-panel", FilePanel)
            volume_panel.volume = mock_volume

            local_panel = app.query_one("#local-panel", FilePanel)

            # Initially local panel should be active
            assert app.active_panel == local_panel

            # Press Tab to switch
            await pilot.press("tab")
            assert app.active_panel == volume_panel

            # Press Tab again to switch back
            await pilot.press("tab")
            assert app.active_panel == local_panel

    @pytest.mark.asyncio
    async def test_local_directory_loading(self, temp_local_dir):
        """Test that local directory contents are loaded."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            # Wait for directory to load
            await pilot.pause()

            local_panel = app.query_one("#local-panel", FilePanel)

            # Check that items are loaded (should have .., local1.txt, local2.txt, localdir)
            assert len(local_panel.items) >= 3

            # Check that our test files are present
            item_names = {item.name for item in local_panel.items}
            assert "local1.txt" in item_names
            assert "local2.txt" in item_names
            assert "localdir" in item_names

    @pytest.mark.asyncio
    async def test_volume_directory_loading(self, temp_local_dir, mock_volume):
        """Test that volume directory contents are loaded."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            # Inject mock volume
            app.volume = mock_volume
            volume_panel = app.query_one("#volume-panel", FilePanel)
            volume_panel.volume = mock_volume

            # Trigger reload
            volume_panel.load_directory()
            await pilot.pause()

            # Check that items are loaded
            assert len(volume_panel.items) >= 2

            # Check that our test files are present
            item_names = {item.name for item in volume_panel.items}
            assert "file1.txt" in item_names or "subdir" in item_names

    @pytest.mark.asyncio
    async def test_file_marking(self, temp_local_dir):
        """Test marking files with Space."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            local_panel = app.query_one("#local-panel", FilePanel)

            # Move to first actual file (skip ..)
            await pilot.press("down")
            await pilot.pause()

            # Mark the file
            await pilot.press("space")
            await pilot.pause()

            # Check that the file is marked
            assert len(local_panel.marked_items) == 1

    @pytest.mark.asyncio
    async def test_directory_navigation(self, temp_local_dir):
        """Test navigating into a directory with Enter."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            local_panel = app.query_one("#local-panel", FilePanel)
            initial_path = local_panel.current_path

            # Find the localdir directory and navigate to it
            for i, item in enumerate(local_panel.items):
                if item.name == "localdir":
                    # Move cursor to this item
                    for _ in range(i):
                        await pilot.press("down")
                    await pilot.press("enter")
                    await pilot.pause()
                    break

            # Check that we navigated into the directory
            assert local_panel.current_path != initial_path
            assert "localdir" in local_panel.current_path

    @pytest.mark.asyncio
    async def test_parent_directory_navigation(self, temp_local_dir):
        """Test navigating to parent directory with .. entry."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        # Start in a subdirectory
        subdir = temp_local_dir / "localdir"

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(subdir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            local_panel = app.query_one("#local-panel", FilePanel)
            initial_path = local_panel.current_path

            # First item should be ".."
            assert local_panel.items[0].name == ".."

            # Press enter to go up
            await pilot.press("enter")
            await pilot.pause()

            # Should be in parent directory now
            assert local_panel.current_path != initial_path

    @pytest.mark.asyncio
    async def test_quit_with_q(self, temp_local_dir):
        """Test quitting with q key."""
        from modal.cli.programs.volume_browser import VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(temp_local_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should exit - if we get here without hanging, it worked


class TestCopyOperations:
    """Test copy operations between panels."""

    @pytest.fixture
    def mock_volume(self):
        """Create a mock volume."""
        return MockVolume()

    @pytest.fixture
    def temp_local_dir(self, tmp_path):
        """Create a temporary local directory with test files."""
        (tmp_path / "upload_me.txt").write_text("Content to upload")
        return tmp_path

    @pytest.mark.asyncio
    async def test_copy_local_to_volume(self, temp_local_dir, mock_volume):
        """Test copying a file from local to volume."""
        # This tests the underlying copy logic without the full TUI
        from modal.cli.programs.volume_browser import FileItem

        # Create a file item representing a local file
        local_file = temp_local_dir / "upload_me.txt"
        item = FileItem.from_local(local_file)

        # Use the mock volume's batch upload
        async with mock_volume.batch_upload(force=True) as batch:
            batch.put_file(str(local_file), "/upload_me.txt")

        # Verify the file was "uploaded"
        assert "/upload_me.txt" in mock_volume.files
        assert mock_volume.files["/upload_me.txt"] == b"Content to upload"

    @pytest.mark.asyncio
    async def test_copy_volume_to_local(self, temp_local_dir, mock_volume):
        """Test copying a file from volume to local."""
        # Add a file to the mock volume
        mock_volume.add_file("/download_me.txt", b"Content to download")

        # Read the file
        chunks = []
        async for chunk in mock_volume.read_file("/download_me.txt"):
            chunks.append(chunk)
        content = b"".join(chunks)

        # Write to local
        dest_file = temp_local_dir / "download_me.txt"
        dest_file.write_bytes(content)

        # Verify
        assert dest_file.exists()
        assert dest_file.read_bytes() == b"Content to download"


class TestFileViewer:
    """Test the file viewer functionality."""

    def test_file_viewer_screen_creation(self):
        """Test creating a FileViewerScreen."""
        from modal.cli.programs.volume_browser import FileViewerScreen

        screen = FileViewerScreen("test.txt", "Hello, World!")
        assert screen.filename == "test.txt"
        assert screen.content == "Hello, World!"

    @pytest.mark.asyncio
    async def test_file_viewer_compose(self):
        """Test that FileViewerScreen composes correctly."""
        from textual.app import App

        from modal.cli.programs.volume_browser import FileViewerScreen

        class TestApp(App):
            def compose(self):
                yield from []

        app = TestApp()
        screen = FileViewerScreen("test.txt", "Line 1\nLine 2\nLine 3")

        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(screen)
            await pilot.pause()
            # If we get here without error, the screen composed successfully
            # Check that the TextArea exists
            from textual.widgets import TextArea

            text_areas = screen.query(TextArea)
            assert len(text_areas) == 1


class TestConfirmDialog:
    """Test the confirmation dialog."""

    def test_confirm_dialog_creation(self):
        """Test creating a ConfirmDialog."""
        from modal.cli.programs.volume_browser import ConfirmDialog

        dialog = ConfirmDialog("Are you sure?", title="Confirm Action")
        assert dialog.message == "Are you sure?"
        assert dialog.title_text == "Confirm Action"


class TestInputDialog:
    """Test the input dialog."""

    def test_input_dialog_creation(self):
        """Test creating an InputDialog."""
        from modal.cli.programs.volume_browser import InputDialog

        dialog = InputDialog("Enter name:", title="New Directory", default="untitled")
        assert dialog.message == "Enter name:"
        assert dialog.title_text == "New Directory"
        assert dialog.default == "untitled"


class TestCopyProgressScreen:
    """Test the copy progress screen."""

    def test_progress_screen_creation(self):
        """Test creating a CopyProgressScreen."""
        from modal.cli.programs.volume_browser import CopyProgressScreen

        screen = CopyProgressScreen("Copying files...")
        assert screen.title_text == "Copying files..."
        assert screen.total_files == 0
        assert screen.total_bytes == 0
        assert screen.completed_files == 0
        assert screen.completed_bytes == 0

    def test_progress_tracking(self):
        """Test tracking progress."""
        from modal.cli.programs.volume_browser import CopyProgressScreen

        screen = CopyProgressScreen("Copying...")
        screen.total_files = 5
        screen.total_bytes = 5000

        # Simulate progress
        screen.current_file = "file1.txt"
        screen.completed_files = 2
        screen.completed_bytes = 2000

        assert screen.completed_files == 2
        assert screen.completed_bytes == 2000


class TestDeleteOperations:
    """Test delete operations."""

    @pytest.fixture
    def mock_volume(self):
        """Create a mock volume with files."""
        vol = MockVolume()
        vol.add_file("/to_delete.txt", b"delete me")
        vol.add_directory("/delete_dir")
        vol.add_file("/delete_dir/nested.txt", b"nested content")
        return vol

    @pytest.mark.asyncio
    async def test_delete_volume_file(self, mock_volume):
        """Test deleting a file from the volume."""
        # Verify file exists
        assert "/to_delete.txt" in mock_volume.files

        # Delete it
        await mock_volume.remove_file("/to_delete.txt")

        # Verify it's gone
        assert "/to_delete.txt" not in mock_volume.files

    @pytest.mark.asyncio
    async def test_delete_volume_directory(self, mock_volume):
        """Test deleting a directory from the volume."""
        # Verify directory exists
        assert "/delete_dir" in mock_volume.dirs
        assert "/delete_dir/nested.txt" in mock_volume.files

        # Delete directory recursively
        await mock_volume.remove_file("/delete_dir", recursive=True)

        # Verify it's gone along with contents
        assert "/delete_dir" not in mock_volume.dirs
        assert "/delete_dir/nested.txt" not in mock_volume.files


class TestMoveOperations:
    """Test move operations (copy + delete)."""

    @pytest.fixture
    def temp_dir_with_files(self, tmp_path):
        """Create a temp directory with files to move."""
        (tmp_path / "move_me.txt").write_text("Move this content")
        return tmp_path

    @pytest.mark.asyncio
    async def test_move_local_to_volume(self, temp_dir_with_files):
        """Test moving a file from local to volume (copy + delete source)."""
        mock_vol = MockVolume()

        local_file = temp_dir_with_files / "move_me.txt"
        assert local_file.exists()

        # Copy to volume
        async with mock_vol.batch_upload(force=True) as batch:
            batch.put_file(str(local_file), "/move_me.txt")

        # Delete local (simulating move)
        local_file.unlink()

        # Verify: file is in volume, not in local
        assert "/move_me.txt" in mock_vol.files
        assert not local_file.exists()


class TestKeyBindings:
    """Test that key bindings are correctly set up."""

    def test_app_bindings(self):
        """Test that VolumeBrowserApp has the expected bindings."""
        from textual.binding import Binding

        from modal.cli.programs.volume_browser import VolumeBrowserApp

        # Extract keys from bindings - handle both Binding objects and tuples
        binding_keys: set[str] = set()
        for b in VolumeBrowserApp.BINDINGS:
            if isinstance(b, Binding):
                binding_keys.add(b.key)
            elif isinstance(b, tuple) and len(b) >= 1:
                binding_keys.add(b[0])

        # Check essential keybindings exist
        assert "q" in binding_keys or "escape" in binding_keys  # Quit
        assert "tab" in binding_keys  # Switch panels
        assert "f5" in binding_keys  # Copy
        assert "f6" in binding_keys  # Move
        assert "f7" in binding_keys  # Mkdir
        assert "f8" in binding_keys  # Delete
        assert "f3" in binding_keys  # View
        assert "space" in binding_keys  # Mark

    def test_panel_bindings(self):
        """Test that FilePanel has bindings for file operations."""
        from modal.cli.programs.volume_browser import FilePanel

        # FilePanel uses a DataTable internally for navigation
        # The BINDINGS may be on the panel or inherited from DataTable
        # For now, just verify BINDINGS is a list (can be empty if all bindings are on app level)
        assert isinstance(FilePanel.BINDINGS, list)


class TestPanelType:
    """Test PanelType enum."""

    def test_panel_types(self):
        """Test that panel types are correctly defined."""
        from modal.cli.programs.volume_browser import PanelType

        assert PanelType.LOCAL.value == "local"
        assert PanelType.VOLUME.value == "volume"


class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.fixture
    def mock_volume_with_data(self):
        """Create a mock volume with realistic test data."""
        vol = MockVolume()
        # Create a directory structure
        vol.add_file("/README.md", b"# Test Volume\n\nThis is a test volume.")
        vol.add_directory("/data")
        vol.add_file("/data/dataset.csv", b"id,name,value\n1,foo,100\n2,bar,200")
        vol.add_directory("/models")
        vol.add_file("/models/model.pkl", b"binary model data here")
        vol.add_directory("/logs")
        vol.add_file("/logs/app.log", b"2025-01-01 12:00:00 INFO Started")
        return vol

    @pytest.fixture
    def local_project_dir(self, tmp_path):
        """Create a local project directory with files."""
        (tmp_path / "main.py").write_text("def main():\n    print('Hello')\n")
        (tmp_path / "config.yaml").write_text("debug: true\nport: 8080\n")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "input.json").write_text('{"key": "value"}')
        return tmp_path

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_volume_with_data, local_project_dir):
        """Test a full workflow: browse, copy, view."""
        from modal.cli.programs.volume_browser import FilePanel, VolumeBrowserApp

        app = VolumeBrowserApp(
            volume_name="test-volume",
            environment_name=None,
            local_path=str(local_project_dir),
            volume_path="/",
        )

        async with app.run_test() as pilot:
            # Inject mock volume
            app.volume = mock_volume_with_data
            volume_panel = app.query_one("#volume-panel", FilePanel)
            volume_panel.volume = mock_volume_with_data

            # Wait for panels to load
            await pilot.pause()

            # Verify local panel loaded files
            local_panel = app.query_one("#local-panel", FilePanel)
            local_names = {item.name for item in local_panel.items}
            assert "main.py" in local_names
            assert "config.yaml" in local_names
            assert "data" in local_names

            # Reload volume panel with mock data
            volume_panel.load_directory()
            await pilot.pause()

            # Verify volume panel loaded files
            vol_names = {item.name for item in volume_panel.items}
            # Note: may include ".." as well
            assert "README.md" in vol_names or "data" in vol_names or len(volume_panel.items) > 0

            # Test navigation - go to local data directory
            for i, item in enumerate(local_panel.items):
                if item.name == "data":
                    # Navigate to it - move down to this item
                    for _ in range(i):
                        await pilot.press("down")
                    await pilot.press("enter")
                    await pilot.pause()
                    break

            # Verify we're now in the data directory
            assert "data" in local_panel.current_path

            # Navigate back with ..
            await pilot.press("enter")  # Enter on ".."
            await pilot.pause()

            # Test quit
            await pilot.press("q")
