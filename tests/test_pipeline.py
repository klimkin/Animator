import io
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from animator.providers.base import ModelNames, Narration
from animator.regions import Box
from animator.storyboard.pipeline import StoryboardOptions, run_storyboard
from animator.storyboard.schemas import Beat, DetectedRegion, DetectionResponse, StoryboardPlan, StoryboardScene

BEAT_SECONDS = 0.8

PLAN = StoryboardPlan(title="Sky", scenes=[
    StoryboardScene(title="Day", layout="sun left, cloud right", beats=[
        Beat(label="SUN", visual="yellow sun", narration="The sun rises."),
        Beat(label="CLOUD", visual="white cloud", narration="A cloud drifts by."),
    ]),
    StoryboardScene(title="Night", layout="moon left, star right", beats=[
        Beat(label="MOON", visual="moon", narration="The moon appears."),
        Beat(label="STAR", visual="star", narration="A star twinkles."),
    ]),
])

DETECTION = DetectionResponse(regions=[
    DetectedRegion(label="left", role="main_concept", object="circle", reveal_order=1,
                   box=Box(ymin=200, xmin=100, ymax=800, xmax=450), reveal="stroke", beat=1),
    DetectedRegion(label="right", role="main_concept", object="square", reveal_order=2,
                   box=Box(ymin=200, xmin=550, ymax=800, xmax=900), reveal="fill", beat=2),
])


def scene_png() -> bytes:
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 80, 280, 280), outline="black", width=6)
    draw.rectangle((380, 90, 560, 270), fill="#202020")
    draw.line((290, 180, 370, 180), fill="black", width=5)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FakeLLM:
    defaults = ModelNames(text="t", region="r", image="i")

    def __init__(self):
        self.calls = []

    def structured(self, model, prompt, schema, image=None):
        self.calls.append((model, schema.__name__, image))
        return {StoryboardPlan: PLAN, DetectionResponse: DETECTION}[schema]

    def image(self, model, prompt):
        self.calls.append((model, "image", None))
        return scene_png()


class FakeNarrator:
    def __init__(self, tone):
        self.tone = tone
        self.calls = []

    def narrate(self, scenes):
        self.calls.append(scenes)
        return [
            Narration(
                wav=self.tone(BEAT_SECONDS * len(beats)),
                windows=[[BEAT_SECONDS * i, BEAT_SECONDS * (i + 1)] for i in range(len(beats))],
            )
            for beats in scenes
        ]


def probe(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout)


def test_run_storyboard_renders_video_paced_to_narration(tmp_path, tone):
    llm, narrator = FakeLLM(), FakeNarrator(tone)
    output = tmp_path / "out.mp4"
    options = StoryboardOptions(
        idea="the sky", output=output, quality="low", models=llm.defaults, save_assets=True,
    )

    durations = run_storyboard(llm, narrator, options)

    assert narrator.calls == [[[beat.narration for beat in scene.beats] for scene in PLAN.scenes]]
    assert sorted(call[:2] for call in llm.calls) == sorted(
        [("t", "StoryboardPlan"), ("i", "image"), ("i", "image"),
         ("r", "DetectionResponse"), ("r", "DetectionResponse")]
    )
    narration_seconds = BEAT_SECONDS * 4
    assert abs(sum(durations) - narration_seconds) < 0.05
    assert abs(probe(output) - narration_seconds) < 0.3
    assets = tmp_path / "out.storyboard"
    assert (assets / "scene_01.regions.json").is_file()
    assert "beat_windows" in (assets / "scene_00.regions.json").read_text()
