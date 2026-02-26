# Copyright Modal Labs 2026
import webbrowser

from modal.output import OutputManager


def open_url(url: str) -> bool:
    """Opens url in web browser, making sure we use a modern one (not Lynx etc)"""
    try:
        browser = webbrowser.get()
        # zpresto defines `BROWSER=open` by default on macOS, which causes `webbrowser` to return `GenericBrowser`.
        if isinstance(browser, webbrowser.GenericBrowser) and browser.name != "open":
            return False
        else:
            return browser.open_new_tab(url)
    except webbrowser.Error:
        return False


def open_url_and_display(url: str, target: str) -> None:
    """Open a URL in the web browser and display it to the user."""
    output = OutputManager.get()
    if open_url(url):
        output.print(f"Opening {target} in your web browser...")
        output.print(f"[link={url}]{url}[/link]")
    else:
        output.print("[yellow]Could not open web browser automatically.[/yellow]")
        output.print("Please open this URL in your browser:")
        output.print(f"[link={url}]{url}[/link]")
