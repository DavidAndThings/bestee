from bestee_chat import __version__
from bestee_chat.main import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_main_runs(capsys) -> None:
    main()
    captured = capsys.readouterr()
    assert "bestee-chat" in captured.out
