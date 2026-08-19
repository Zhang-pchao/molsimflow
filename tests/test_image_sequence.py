from molsimflow.media.image_sequence import discover_numbered_images, make_even_crop


def test_numbered_image_discovery_has_no_fixed_prefix_or_start_index(tmp_path):
    for name in ("render_12.png", "render_10.png", "render_11.png", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")

    files = discover_numbered_images(tmp_path, "*.png")

    assert [path.name for path in files] == ["render_10.png", "render_11.png", "render_12.png"]
    assert make_even_crop((0, 0, 11, 9), (11, 9)) == (0, 0, 10, 8)
