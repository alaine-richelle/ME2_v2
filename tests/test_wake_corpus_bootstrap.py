import json

from me2.wake_word import corpus_bootstrap


def test_bootstrap_creates_static_templates_and_only_aggregate_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_bootstrap, "PRIVATE_ROOT", tmp_path.resolve())
    report = corpus_bootstrap.bootstrap(tmp_path)
    template_directory = tmp_path / "templates"
    assert report["root_ready_for_human_consent_registration"]
    assert report["templates_created"] == 2
    assert report["inventory_before"] == {"files": 0, "directories": 0}
    assert sorted(path.name for path in template_directory.iterdir()) == [
        "consent-manifest.schema.json",
        "derived-manifest.schema.json",
    ]
    assert "spk-" not in json.dumps(report)


def test_bootstrap_preserves_existing_templates(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_bootstrap, "PRIVATE_ROOT", tmp_path.resolve())
    corpus_bootstrap.bootstrap(tmp_path)
    report = corpus_bootstrap.bootstrap(tmp_path)
    assert report["templates_created"] == 0
    assert report["templates_already_present"] == 2
