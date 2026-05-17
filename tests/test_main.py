"""Tests for bestee.main."""

import pytest

from bestee.main import main


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that main prints the expected greeting."""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from bestee!\n"
