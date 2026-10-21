"""In-page scripts from dom.py, run in Node against a minimal fake DOM."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from muse2api.drivers.browser import dom

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

# The fake document answers querySelectorAll with the elements listed under the
# exact selector string; bubbles with inButton=true sit inside a quick-reply button.
FAKE_DOM = """
const mk = (b) => ({
  innerText: b.text, outerHTML: '<div>' + b.text + '</div>',
  closest: (sel) => (sel === QUICK_REPLY && b.inButton ? {} : null),
  querySelectorAll: () => [], querySelector: () => null,
});
const bySelector = { [AGENT_BUBBLE]: BUBBLES.map(mk) };
globalThis.getComputedStyle = () => ({ overflowY: 'visible' });
globalThis.document = {
  readyState: 'complete',
  body: { innerText: BUBBLES.map(b => b.text).join('\\n') },
  querySelectorAll: (sel) => bySelector[sel] || [],
  querySelector: (sel) => (bySelector[sel] || [])[0] || null,
};
"""


def run(script: str, bubbles: list[dict]):
    prelude = (f"const AGENT_BUBBLE = {dom.q(dom.AGENT_BUBBLE)};"
               f"const QUICK_REPLY = {dom.q(dom.QUICK_REPLY)};"
               f"const BUBBLES = {json.dumps(bubbles)};")
    code = prelude + FAKE_DOM + f"process.stdout.write(JSON.stringify({script}));"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


# What muse.ai renders after a reply that offers quick-reply suggestions.
WITH_SUGGESTIONS = [
    {"text": "Hey! I'm your personal agent."},
    {"text": "Want a new look? Here's a link: https://muse.ai/files/x.png"},
    {"text": "Give me sunglasses", "inButton": True},
]


def test_chat_state_ignores_quick_reply_buttons():
    state = run(dom.CHAT_STATE, WITH_SUGGESTIONS)
    assert state["agentCount"] == 2
    assert state["lastText"] == WITH_SUGGESTIONS[1]["text"]


def test_last_bubble_media_reads_the_reply_not_the_suggestion():
    probe = run(dom.LAST_BUBBLE_MEDIA, WITH_SUGGESTIONS)
    assert probe["links"] == ["https://muse.ai/files/x.png"]
