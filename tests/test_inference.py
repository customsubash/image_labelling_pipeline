import os

from b_inference import run_inference

def test_inference_mock(monkeypatch):
    called = []
    def mock_post(url, files):
        called.append(True)
        class Resp:
            def json(self):
                return {"bboxes": [{"category_id": 0, "bbox": [0, 0, 10, 10]}], "width": 100, "height": 100}
        return Resp()
    monkeypatch.setattr("requests.post", mock_post)
    preds = run_inference("tests/dummy_input")
    assert isinstance(preds, list)
    assert called
