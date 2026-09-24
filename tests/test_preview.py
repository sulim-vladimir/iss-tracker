"""The page is built from files in issctl/web, so a typo there breaks it at request time.

These exist because the templates used to live inside a Python .format() string, where every
CSS and JavaScript brace had to be doubled - a rule that was broken three times in one evening.
"""

import re
import urllib.request

import numpy as np
import pytest

from issctl.preview import Preview, asset, fill, fragments


class _Cam:
    bayer = False
    exposure_ms, gain, fps = 50.0, 100, 5

    def __init__(self, name, w, h):
        self.name, self.width, self.height = name, w, h

    def latest(self):
        return np.zeros((self.height, self.width), np.uint8), None, 1


def _preview(controls=True, port=None):
    p = Preview.__new__(Preview)
    p.cams = {"guide": _Cam("guide", 1280, 960), "main": _Cam("main", 1936, 1096)}
    p.state, p.status, p.port = {}, None, port
    p.controls = ({"mount_action": lambda *a: None, "record": lambda *a: None,
                   "estop": lambda: None, "sky": lambda: None} if controls else None)
    return p


def test_page_has_no_unfilled_placeholders():
    """A {{name}} left in the output means a fragment moved and its value did not follow."""
    html = _preview().page()
    assert "{{" not in html and "}}" not in html


def test_page_renders_without_any_controls():
    """The preview also runs read-only, where every control fragment is empty."""
    assert "<body>" in _preview(controls=False).page()


def test_fill_refuses_a_placeholder_it_was_not_given():
    """Silently leaving {{mount_panel}} in the page would look like a layout bug, not a typo."""
    with pytest.raises(KeyError):
        fill("<p>{{nobody}}</p>")


def test_every_fragment_the_page_needs_exists():
    f = fragments()
    for name in ("panel", "centre", "in_frame", "mount", "status", "warnings", "messages",
                 "record", "estop", "sky"):
        assert name in f and f[name].strip(), name


def test_javascript_only_touches_elements_the_page_contains():
    """getElementById on a missing node throws and silently freezes every other readout.

    The sky chart is the deliberate exception: it is parked out of the page, and drawSky returns
    early when its svg is absent, so its two lookups can never run.
    """
    html = _preview().page()
    js = asset("app.js")
    start = js.index("function drawSky")
    depth, end = 0, start
    for i in range(js.index("{", start), len(js)):
        depth += (js[i] == "{") - (js[i] == "}")
        if depth == 0:
            end = i
            break
    outside = js[:start] + js[end:]
    missing = [i for i in set(re.findall(r"getElementById\('([a-z-]+)'\)", outside))
               if f'id="{i}"' not in html]
    assert not missing, missing


def test_stylesheet_and_script_are_served():
    p = _preview(port=8098)
    server = p.start()
    try:
        get = lambda path: urllib.request.urlopen(f"http://127.0.0.1:8098{path}", timeout=5)
        assert get("/style.css").headers["Content-Type"] == "text/css"
        assert get("/app.js").headers["Content-Type"] == "application/javascript"
        assert b"<html" in get("/").read()
    finally:
        server.shutdown()
