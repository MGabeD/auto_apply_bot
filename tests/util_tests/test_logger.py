from auto_apply_bot.utils.logger import get_logger


def test_logger_creates_log_file_and_writes(tmp_path, monkeypatch):
    """
    Test that the logger creates a log file and writes to it. Seems unneccesary to test this. But since it is kinda core to everything being debuggable,
    I thought it would be good to test.
    """
    monkeypatch.setattr("auto_apply_bot.utils.logger.resolve_project_source", lambda: tmp_path)

    logger = get_logger("test_logger")
    logger.info("Hello test logger!")

    logs_dir = tmp_path / "logs"
    log_files = list(logs_dir.glob("pipeline_run_*.log"))
    assert len(log_files) == 1

    with open(log_files[0], "r") as f:
        content = f.read()
        assert "Hello test logger!" in content
