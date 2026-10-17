"""STEP 5 orchestrator for --prompt: idea -> storyboard -> scene images and
narration -> regions -> one rendered video. Start reading here."""
from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..providers.base import LLM, ModelNames, Narrator
from ..render import Scene, render_video
from .detect import detect_region_plan
from .images import generate_scene_image
from .schemas import StoryboardScene
from .script import generate_storyboard


@dataclass
class StoryboardOptions:
    idea: str
    output: Path
    quality: str
    models: ModelNames
    num_scenes: int | None = None
    save_assets: bool = False


async def detect_scene(
    llm: LLM, options: StoryboardOptions, image_path: Path, audio_path: Path,
    index: int, scene: StoryboardScene, beat_windows: list[list[float]],
) -> Scene:
    region_plan = await asyncio.to_thread(
        detect_region_plan,
        llm,
        image_path,
        scene.narration,
        scene.title,
        options.models.region,
        [beat.narration for beat in scene.beats],
    )
    region_plan.beat_windows = beat_windows
    if options.save_assets:
        region_path = image_path.with_name(f"scene_{index:02d}.regions.json")
        region_path.write_text(region_plan.model_dump_json(indent=2))
    print(f"  scene {index + 1} ready: {scene.title}", file=sys.stderr)
    return Scene(image=str(image_path), audio=str(audio_path), region_plan=region_plan)


async def render_storyboard(
    llm: LLM, narrator: Narrator, options: StoryboardOptions, asset_dir: Path,
) -> list[float]:
    # STEP 1
    print(f"generating storyboard for: {options.idea!r}", file=sys.stderr)
    plan = generate_storyboard(llm, options.idea, model=options.models.text, num_scenes=options.num_scenes)
    print(f"storyboard: {len(plan.scenes)} scene(s)", file=sys.stderr)
    if options.save_assets:
        (asset_dir / "storyboard.json").write_text(plan.model_dump_json(indent=2))

    # STEPS 2-3: every image in parallel, alongside one narrate() call for the whole video
    images, narrations = await asyncio.gather(
        asyncio.gather(*(
            asyncio.to_thread(generate_scene_image, llm, scene, options.models.image) for scene in plan.scenes
        )),
        asyncio.to_thread(narrator.narrate, [[beat.narration for beat in scene.beats] for scene in plan.scenes]),
    )
    detections = []
    for index, (scene, image_bytes, narration) in enumerate(zip(plan.scenes, images, narrations, strict=True)):
        image_path = asset_dir / f"scene_{index:02d}.png"
        image_path.write_bytes(image_bytes)
        audio_path = asset_dir / f"scene_{index:02d}.wav"
        audio_path.write_bytes(narration.wav)
        detections.append(detect_scene(llm, options, image_path, audio_path, index, scene, narration.windows))

    # STEP 4
    scenes = await asyncio.gather(*detections)
    # STEP 5
    return render_video(scenes, str(options.output), quality=options.quality)


def run_storyboard(llm: LLM, narrator: Narrator, options: StoryboardOptions) -> list[float]:
    """Keep assets next to the output when asked, otherwise in a temp dir."""
    if options.save_assets:
        asset_dir = options.output.parent / f"{options.output.stem}.storyboard"
        asset_dir.mkdir(parents=True, exist_ok=True)
        return asyncio.run(render_storyboard(llm, narrator, options, asset_dir))
    with tempfile.TemporaryDirectory() as temp_dir:
        return asyncio.run(render_storyboard(llm, narrator, options, Path(temp_dir)))
