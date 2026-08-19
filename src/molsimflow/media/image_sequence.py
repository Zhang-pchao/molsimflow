"""Convert a naturally numbered image sequence into an optional annotated MP4."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_FRAME_INDEX_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def discover_numbered_images(
    image_dir: Path,
    pattern: str = "*.bmp",
    *,
    require_contiguous: bool = True,
) -> list[Path]:
    """Return image files sorted by their final numeric filename component."""

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    indexed = []
    for path in image_dir.glob(pattern):
        match = _FRAME_INDEX_RE.search(path.name)
        if path.is_file() and match is not None:
            indexed.append((int(match.group(1)), path.name, path))
    if not indexed:
        raise FileNotFoundError(f"No numbered images matching {pattern!r} in {image_dir}")
    indexed.sort()
    indices = [item[0] for item in indexed]
    if len(indices) != len(set(indices)):
        raise ValueError("Image sequence contains duplicate numeric frame indices")
    if require_contiguous and indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"Image frame indices are not contiguous: {indices}")
    return [item[2] for item in indexed]


def content_bbox(image: Any, white_threshold: int = 250) -> tuple[int, int, int, int]:
    """Return the non-white bounding box as a PIL half-open crop rectangle."""

    if not 0 <= white_threshold <= 255:
        raise ValueError("white_threshold must be between 0 and 255")
    pixels = np.asarray(image.convert("RGB"))
    mask = np.any(pixels < white_threshold, axis=2)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Reference image contains no pixels below the white threshold")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def make_even_crop(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Adjust a crop to even dimensions required by common H.264 pixel formats."""

    left, top, right, bottom = bbox
    width, height = image_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Crop box must lie inside the image")
    if (right - left) % 2:
        if right < width:
            right += 1
        elif left > 0:
            left -= 1
        else:
            right -= 1
    if (bottom - top) % 2:
        if bottom < height:
            bottom += 1
        elif top > 0:
            top -= 1
        else:
            bottom -= 1
    if right - left < 2 or bottom - top < 2:
        raise ValueError("Video crop must be at least 2 x 2 pixels")
    return left, top, right, bottom


def _load_font(font_size: int, font_path: Path | None) -> Any:
    from PIL import ImageFont

    if font_path is not None:
        if not Path(font_path).is_file():
            raise FileNotFoundError(font_path)
        return ImageFont.truetype(str(font_path), font_size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _annotate(image: Any, text: str, font: Any) -> Any:
    from PIL import ImageDraw

    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    margin = max(8, output.width // 80)
    position = (output.width - (right - left) - margin, output.height - (bottom - top) - margin)
    draw.text(position, text, fill="black", font=font, stroke_width=2, stroke_fill="white")
    return output


def make_video(
    image_dir: Path,
    output_path: Path,
    *,
    pattern: str = "*.bmp",
    reference_image: str | None = None,
    crop_white_border: bool = False,
    white_threshold: int = 250,
    cropped_frames_dir: Path | None = None,
    fps: float = 5.0,
    time_start: float = 0.0,
    time_step: float | None = None,
    time_end: float | None = None,
    time_unit: str = "ps",
    font_path: Path | None = None,
    font_size: int | None = None,
    require_contiguous: bool = True,
) -> dict[str, object]:
    """Write an H.264 MP4 from an image sequence.

    Time labels are omitted unless ``time_step`` or ``time_end`` is supplied.
    ``time_step`` and ``time_end`` are mutually exclusive.
    """

    import imageio.v2 as imageio
    from PIL import Image

    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive")
    if time_step is not None and time_end is not None:
        raise ValueError("Provide time_step or time_end, not both")
    if not math.isfinite(time_start):
        raise ValueError("time_start must be finite")
    if time_step is not None and not math.isfinite(time_step):
        raise ValueError("time_step must be finite")
    if time_end is not None and not math.isfinite(time_end):
        raise ValueError("time_end must be finite")
    if font_size is not None and font_size <= 0:
        raise ValueError("font_size must be positive")
    files = discover_numbered_images(
        image_dir, pattern=pattern, require_contiguous=require_contiguous
    )
    reference_path = files[0] if reference_image is None else Path(image_dir) / reference_image
    if reference_path not in files:
        raise FileNotFoundError(
            f"Reference image is not in the selected sequence: {reference_path}"
        )
    with Image.open(reference_path) as reference:
        reference_rgb = reference.convert("RGB")
        crop = (
            content_bbox(reference_rgb, white_threshold)
            if crop_white_border
            else (0, 0, *reference_rgb.size)
        )
        crop = make_even_crop(crop, reference_rgb.size)
        reference_size = reference_rgb.size

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_path must use the .mp4 extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    if cropped_frames_dir is not None:
        cropped_frames_dir = Path(cropped_frames_dir).expanduser().resolve()
        cropped_frames_dir.mkdir(parents=True, exist_ok=True)
    annotate = time_step is not None or time_end is not None
    size = (crop[2] - crop[0], crop[3] - crop[1])
    font = _load_font(font_size or max(18, min(size) // 24), font_path) if annotate else None

    temporary = output.with_name(output.stem + ".tmp.mp4")
    writer = None
    try:
        writer = imageio.get_writer(
            str(temporary),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=1,
            pixelformat="yuv420p",
            ffmpeg_params=["-movflags", "+faststart"],
        )
        for index, path in enumerate(files):
            with Image.open(path) as source:
                image = source.convert("RGB")
                if image.size != reference_size:
                    raise ValueError(
                        f"Image size changed at {path.name}: {image.size} vs {reference_size}"
                    )
                image = image.crop(crop)
                if cropped_frames_dir is not None:
                    image.save(cropped_frames_dir / f"frame_{index:06d}.png")
                if annotate:
                    value = (
                        time_start + float(time_step) * index
                        if time_step is not None
                        else time_start
                        + (float(time_end) - time_start) * index / max(len(files) - 1, 1)
                    )
                    image = _annotate(image, f"{value:.2f} {time_unit}", font)
                writer.append_data(np.asarray(image, dtype=np.uint8))
        writer.close()
        writer = None
        temporary.replace(output)
    except BaseException:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
        raise

    metadata_path = output.with_name(output.name + ".json")
    metadata = {
        "workflow": "image_sequence_video",
        "input_directory": str(Path(image_dir).expanduser().resolve()),
        "pattern": pattern,
        "reference_image": str(reference_path),
        "output": str(output),
        "metadata": str(metadata_path),
        "frame_count": len(files),
        "fps": fps,
        "duration_seconds": len(files) / fps,
        "crop_box_xyxy": crop,
        "output_frame_size": size,
        "time_start": time_start if annotate else None,
        "time_step": time_step,
        "time_end": time_end,
        "time_unit": time_unit if annotate else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.bmp")
    parser.add_argument("--reference-image")
    parser.add_argument("--crop-white-border", action="store_true")
    parser.add_argument("--white-threshold", type=int, default=250)
    parser.add_argument("--cropped-frames-dir", type=Path)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--time-start", type=float, default=0.0)
    time = parser.add_mutually_exclusive_group()
    time.add_argument("--time-step", type=float)
    time.add_argument("--time-end", type=float)
    parser.add_argument("--time-unit", default="ps")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-size", type=int)
    parser.add_argument("--allow-gaps", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = make_video(
        args.image_dir,
        args.output,
        pattern=args.pattern,
        reference_image=args.reference_image,
        crop_white_border=args.crop_white_border,
        white_threshold=args.white_threshold,
        cropped_frames_dir=args.cropped_frames_dir,
        fps=args.fps,
        time_start=args.time_start,
        time_step=args.time_step,
        time_end=args.time_end,
        time_unit=args.time_unit,
        font_path=args.font,
        font_size=args.font_size,
        require_contiguous=not args.allow_gaps,
    )
    print(metadata["output"])
    print(metadata["metadata"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
