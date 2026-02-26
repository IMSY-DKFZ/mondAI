"""Test the mondAI package's __version__ attribute."""


def test_version() -> None:
    from mondAI import __version__

    assert isinstance(__version__, str)
