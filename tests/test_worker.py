import importlib


def test_dummy():
    assert True


def test_import():
    """Import our newly created module, through importlib to avoid parsing issues"""
    worker = importlib.import_module("worker_olmocr.worker")
    assert hasattr(worker, "Demo")
    assert hasattr(worker.Demo, "process_element")
