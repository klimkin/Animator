"""STEP 5 orchestrator for --prompt: idea -> storyboard -> (scene images ->
regions) alongside narration -> one rendered video. Start reading here."""
from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..providers.base import LLM, ModelNames, Narrator
from ..regions import SnippetRegionPlan
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


async def draw_scene(
    llm: LLM, options: StoryboardOptions, asset_dir: Path, index: int, scene: StoryboardScene,
) -> tuple[Path, SnippetRegionPlan]:
    """Image, then regions: detection needs the image and the beat text, not the audio."""
    image_path = asset_dir / f"scene_{index:02d}.png"
    image_path.write_bytes(await asyncio.to_thread(generate_scene_image, llm, scene, options.models.image))
    region_plan = await asyncio.to_thread(
        detect_region_plan,
        llm,
        image_path,
        scene.narration,
        scene.title,
        options.models.region,
        [beat.narration for beat in scene.beats],
    )
    print(f"  scene {index + 1} drawn: {scene.title}", file=sys.stderr)
    return image_path, region_plan


async def render_storyboard(
    llm: LLM, narrator: Narrator, options: StoryboardOptions, asset_dir: Path,
) -> list[float]:
    # STEP 1
    print(f"generating storyboard for: {options.idea!r}", file=sys.stderr)
    plan = generate_storyboard(llm, options.idea, model=options.models.text, num_scenes=options.num_scenes)
    print(f"storyboard: {len(plan.scenes)} scene(s)", file=sys.stderr)
    if options.save_assets:
        (asset_dir / "storyboard.json").write_text(plan.model_dump_json(indent=2))

    # STEPS 2-4: every scene's image and regions in parallel, alongside one narrate() call for the whole video
    drawn, narrations = await asyncio.gather(
        asyncio.gather(*(draw_scene(llm, options, asset_dir, index, scene) for index, scene in enumerate(plan.scenes))),
        asyncio.to_thread(narrator.narrate, [[beat.narration for beat in scene.beats] for scene in plan.scenes]),
    )
    scenes = []
    for index, ((image_path, region_plan), narration) in enumerate(zip(drawn, narrations, strict=True)):
        audio_path = asset_dir / f"scene_{index:02d}.wav"
        audio_path.write_bytes(narration.wav)
        region_plan.beat_windows = narration.windows
        if options.save_assets:
            (asset_dir / f"scene_{index:02d}.regions.json").write_text(region_plan.model_dump_json(indent=2))
        scenes.append(Scene(image=str(image_path), audio=str(audio_path), region_plan=region_plan))
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
