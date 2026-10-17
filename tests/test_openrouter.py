import base64
import json

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel

from animator.providers.openrouter import API_URL, OpenRouterLLM

KEY = "sk-or-secret-key"


class Answer(BaseModel):
    word: str
    count: int


def llm_with(handler) -> OpenRouterLLM:
    return OpenRouterLLM(KEY, http=httpx.Client(transport=httpx.MockTransport(handler)), retry_delay=0)


def message(**fields) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", **fields}}]})


def test_structured_sends_schema_and_image_and_parses_reply(tmp_path):
    image = tmp_path / "scene.png"
    Image.new("RGB", (8, 8), "white").save(image, format="JPEG")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return message(content='{"word": "sun", "count": 3}')

    answer = llm_with(handler).structured("some/model", "count suns", Answer, image=image)

    assert answer == Answer(word="sun", count=3)
    (request,) = requests
    assert str(request.url) == API_URL
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "some/model"
    assert body["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()
    text, picture = body["messages"][0]["content"]
    assert text == {"type": "text", "text": "count suns"}
    assert picture["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(picture["image_url"]["url"].split(",", 1)[1]) == image.read_bytes()


def test_structured_retries_once_with_the_validation_error():
    replies = iter(['{"word": "sun"}', '{"word": "sun", "count": 1}'])
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return message(content=next(replies))

    assert llm_with(handler).structured("m", "p", Answer) == Answer(word="sun", count=1)
    assert len(bodies) == 2
    correction = bodies[1]["messages"]
    assert correction[1] == {"role": "assistant", "content": '{"word": "sun"}'}
    assert "count" in correction[2]["content"]


def test_structured_gives_up_after_one_correction():
    with pytest.raises(RuntimeError, match="does not match Answer"):
        llm_with(lambda request: message(content="not json")).structured("m", "p", Answer)


def test_image_decodes_data_url():
    png = b"\x89PNG fake bytes"
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return message(content="", images=[{"type": "image_url", "image_url": {"url": url}}])

    assert llm_with(handler).image("img/model", "draw") == png
    assert bodies[0]["modalities"] == ["image", "text"]
    assert bodies[0]["image_config"] == {"aspect_ratio": "16:9"}


def test_image_without_images_fails():
    with pytest.raises(RuntimeError, match="no image"):
        llm_with(lambda request: message(content="sorry")).image("m", "draw")


@pytest.mark.parametrize("failure", [
    httpx.Response(429, text="slow down"),
    httpx.Response(503, text="busy"),
    httpx.ConnectError("reset"),
])
def test_retries_rate_limits_server_errors_and_dropped_connections(failure):
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return message(content='{"word": "a", "count": 2}')

    assert llm_with(handler).structured("m", "p", Answer).count == 2
    assert len(attempts) == 2


def test_client_error_fails_fast_without_leaking_key():
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(401, json={"error": {"message": f"invalid key {KEY}"}})

    with pytest.raises(RuntimeError) as caught:
        llm_with(handler).structured("m", "p", Answer)
    assert len(attempts) == 1
    assert "401" in str(caught.value)
    assert KEY not in str(caught.value)


def test_gives_up_after_max_retries():
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(429, text="slow down")

    llm = OpenRouterLLM(KEY, http=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=3, retry_delay=0)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        llm.image("m", "draw")
    assert len(attempts) == 3
