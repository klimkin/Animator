"""STEP 2: draw one storyboard scene as a 16:9 whiteboard image."""
from __future__ import annotations

from ..prompts import load_prompt
from ..providers.base import LLM
from .schemas import StoryboardScene


def build_image_prompt(scene: StoryboardScene) -> str:
    elements = "\n".join(
        f'{index}. {beat.visual.strip()} — caption "{beat.label.strip().upper()}"'
        for index, beat in enumerate(scene.beats, start=1)
    )
    frame = load_prompt("scene_image").format(
        title=scene.title.strip().upper(),
        count=len(scene.beats),
        elements=elements,
        layout=scene.layout.strip(),
    )
    return f"{frame}\n{load_prompt('image_style')}"


def generate_scene_image(llm: LLM, scene: StoryboardScene, model: str) -> bytes:
    """Render one frame of separate flat doodle icons. Returns image bytes."""
    return llm.image(model, build_image_prompt(scene))
