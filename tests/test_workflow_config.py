from molsimflow.config.workflow import load_workflow_config, shell_export_lines, summarize_config


def test_workflow_config_loads_sections_and_resolves_paths(tmp_path):
    config_path = tmp_path / "workflow.ini"
    config_path.write_text(
        "\n".join(
            [
                "[scheduler]",
                "PYTHON_BIN = python",
                "ENV_SETUP_SCRIPT = setup/env.sh",
                "",
                "[structure]",
                "OUTPUT_DIR = outputs/caseA",
            ]
        ),
        encoding="utf-8",
    )

    config = load_workflow_config(config_path)

    assert config.get("scheduler", "PYTHON_BIN") == "python"
    assert config.resolve_path("structure", "OUTPUT_DIR") == tmp_path / "outputs" / "caseA"


def test_workflow_config_shell_exports_selected_sections(tmp_path):
    config_path = tmp_path / "workflow.ini"
    config_path.write_text(
        "\n".join(
            [
                "[scheduler]",
                "PYTHON_BIN = python",
                "PACKMOL_COMMAND = packmol -seed 1",
                "",
                "[notes]",
                "not-exportable-name = value",
            ]
        ),
        encoding="utf-8",
    )

    config = load_workflow_config(config_path)
    lines = shell_export_lines(config, section_names=["scheduler"])

    assert "export PYTHON_BIN=python" in lines
    assert "export PACKMOL_COMMAND='packmol -seed 1'" in lines


def test_workflow_config_summary(tmp_path):
    config_path = tmp_path / "workflow.ini"
    config_path.write_text("[section]\nKEY = value\n", encoding="utf-8")

    rows = summarize_config(load_workflow_config(config_path))

    assert rows == [{"section": "section", "n_keys": 1, "keys": "KEY"}]
