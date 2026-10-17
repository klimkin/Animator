"""STEP 4: detect a region plan (what to draw, where, and when) from a
finished whiteboard image."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..prompts import load_prompt
from ..providers.base import LLM
from ..regions import Region, SnippetRegionPlan
from .schemas import DetectedRegion, DetectionResponse


def build_region_prompt(transcript_text: str, idea: str, beats: list[str] | None = None) -> str:
    prompt = load_prompt("region_detection").format(idea=idea, transcript_text=transcript_text)
    if not beats:
        return prompt
    numbered = "\n".join(f"  {index}. {text}" for index, text in enumerate(beats, start=1))
    return f"{prompt}\n\n{load_prompt('region_beats').format(numbered=numbered)}"


def to_region(detected: DetectedRegion) -> Region:
    return Region(
        label=detected.label,
        role=detected.role,
        object=detected.object,
        reveal_order=detected.reveal_order,
        box=detected.box,
        expected_visual=detected.object,
        annotation=detected.annotation,
        reveal=detected.reveal,
        beat=detected.beat or None,
    )


def detect_region_plan(
    llm: LLM,
    image_path: Path,
    transcript_text: str,
    idea: str,
    model: str,
    beats: list[str] | None = None,
) -> SnippetRegionPlan:
    idea = idea or "(untitled)"
    with Image.open(image_path) as image:
        size = list(image.size)

    detection = llm.structured(
        model, build_region_prompt(transcript_text, idea, beats), DetectionResponse, image=image_path,
    )
    if not detection.regions:
        raise RuntimeError("Region detection returned no regions.")

    return SnippetRegionPlan(
        idea=idea,
        global_style_notes="",
        regions=[to_region(detected) for detected in detection.regions],
        image_size=size,
    )
