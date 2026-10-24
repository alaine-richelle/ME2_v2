import json

import pytest

from me2.wake_word import corpus_bootstrap, ingestion_audit
from me2.wake_word.corpus_consent import SCHEMA_VERSION, TRAINING_PARTITIONS


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
    assert report["templates_refreshed"] == 0


def test_bootstrap_refreshes_only_a_modified_static_template(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_bootstrap, "PRIVATE_ROOT", tmp_path.resolve())
    corpus_bootstrap.bootstrap(tmp_path)
    template = tmp_path / "templates" / "consent-manifest.schema.json"
    template.write_text("{}\n", encoding="utf-8")
    report = corpus_bootstrap.bootstrap(tmp_path)
    assert report["templates_created"] == 0
    assert report["templates_refreshed"] == 1


def test_verify_templates_reports_only_aggregate_success(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_bootstrap, "PRIVATE_ROOT", tmp_path.resolve())
    corpus_bootstrap.bootstrap(tmp_path)
    report = corpus_bootstrap.verify_templates(tmp_path)
    assert report == {
        "bootstrap_schema_version": 1,
        "templates_verified": 2,
        "templates_valid": True,
    }


def test_verify_templates_rejects_a_modified_static_template(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_bootstrap, "PRIVATE_ROOT", tmp_path.resolve())
    corpus_bootstrap.bootstrap(tmp_path)
    template = tmp_path / "templates" / "consent-manifest.schema.json"
    template.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        corpus_bootstrap.verify_templates(tmp_path)


def test_bootstrap_schema_templates_match_the_fail_closed_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_bootstrap, "PRIVATE_ROOT", tmp_path.resolve())
    corpus_bootstrap.bootstrap(tmp_path)
    template_directory = tmp_path / "templates"
    consent = json.loads((template_directory / "consent-manifest.schema.json").read_text())
    derived = json.loads((template_directory / "derived-manifest.schema.json").read_text())

    assert consent["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert consent["properties"]["speakers"]["minItems"] == len(TRAINING_PARTITIONS)
    assert consent["properties"]["speakers"]["items"]["additionalProperties"] is False
    assert consent["properties"]["speakers"]["items"]["properties"]["partition"]["enum"] == sorted(
        TRAINING_PARTITIONS
    )
    assert derived["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    record = derived["$defs"]["record"]
    assert set(record["required"]) == ingestion_audit._REQUIRED_RECORD
    assert record["properties"]["review_status"] == {"const": "accepted"}
    assert record["properties"]["connected_delivery_status"]["enum"] == [
        "accepted",
        "not_applicable",
    ]
