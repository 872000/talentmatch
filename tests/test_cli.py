"""CLI regression tests for TalentMatch."""

import pytest

from talentmatch.cli import main


def test_main_unknown_resume_name_prints_helpful_error(capsys):
    code = main(["match", "no_such_resume_xyz", "backend_java_jd"])
    assert code == 1
    err = capsys.readouterr().err
    assert "could not find" in err
    assert "no_such_resume_xyz" in err
    assert "backend_java" in err  # points at a bundled example name
