"""Record the README demo video by driving the running web app with Playwright.

Prerequisites: the web app is built (cd frontend && npm run build) and the API is running
(uvicorn api:app --port 8000), which serves the built app at http://127.0.0.1:8000.

    pip install playwright imageio-ffmpeg && python -m playwright install chromium
    python scripts/record_demo.py          # -> docs/demo.mp4 and docs/demo.gif

Each segment is recorded as its own WebM; long waits for the model are sped up when the
segments are joined, and the result is labelled as sped up in the README.
"""

import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
RAW = ROOT / "data" / "recording"
URL = "http://127.0.0.1:8000"
SIZE = {"width": 1280, "height": 720}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# A visible cursor, click ripples and a caption pill: headless recordings show none of these.
OVERLAY_JS = """
() => {
  if (window.__demo) return;
  window.__demo = true;
  const style = document.createElement('style');
  style.textContent = `
    #demo-cursor { position: fixed; z-index: 99999; width: 18px; height: 18px; margin: -3px 0 0 -3px;
      pointer-events: none; transition: transform .08s; }
    #demo-cursor svg { filter: drop-shadow(0 1px 2px rgb(0 0 0 / .5)); }
    .demo-ripple { position: fixed; z-index: 99998; width: 34px; height: 34px; margin: -17px 0 0 -17px;
      border-radius: 50%; border: 2px solid #7b96ff; pointer-events: none; animation: demo-r .5s ease-out forwards; }
    @keyframes demo-r { from { transform: scale(.3); opacity: 1 } to { transform: scale(1.4); opacity: 0 } }
    #demo-caption { position: fixed; z-index: 99997; left: 50%; bottom: 92px; transform: translateX(-50%);
      background: rgb(20 22 34 / .92); color: #fff; font: 600 17px system-ui, sans-serif; padding: 10px 18px;
      border-radius: 999px; border: 1px solid rgb(123 150 255 / .5); box-shadow: 0 6px 24px rgb(0 0 0 / .35);
      transition: opacity .35s; opacity: 0; pointer-events: none; white-space: nowrap; }`;
  document.head.appendChild(style);
  const c = document.createElement('div');
  c.id = 'demo-cursor';
  c.innerHTML = '<svg width="18" height="22" viewBox="0 0 18 22"><path d="M1 1l15 11-7 1-3 7z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  document.body.appendChild(c);
  const cap = document.createElement('div');
  cap.id = 'demo-caption';
  document.body.appendChild(cap);
  addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', e => {
    const r = document.createElement('div'); r.className = 'demo-ripple';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
    document.body.appendChild(r); setTimeout(() => r.remove(), 600);
  }, true);
  window.__caption = t => { cap.textContent = t; cap.style.opacity = t ? 1 : 0; };
}
"""


class Demo:
    def __init__(self, page: Page):
        self.page = page
        self.t0 = time.time()
        self.marks: dict[str, float] = {}
        self.mouse = (640, 360)

    def mark(self, name: str) -> None:
        self.marks[name] = round(time.time() - self.t0, 2)

    def pause(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))

    def caption(self, text: str, hold: float = 0) -> None:
        self.page.evaluate("t => window.__caption(t)", text)
        if hold:
            self.pause(hold)

    def move_to(self, selector: str, nth: int = 0) -> None:
        box = self.page.locator(selector).nth(nth).bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.mouse.move(x, y, steps=25)
        self.mouse = (x, y)

    def click(self, selector: str, nth: int = 0) -> None:
        self.move_to(selector, nth)
        self.pause(0.25)
        self.page.mouse.down()
        self.page.mouse.up()

    def type(self, selector: str, text: str) -> None:
        self.click(selector)
        self.page.keyboard.type(text, delay=45)


def open_page(browser, video_dir: Path):
    ctx = browser.new_context(
        viewport=SIZE, record_video_dir=str(video_dir), record_video_size=SIZE, color_scheme="dark"
    )
    page = ctx.new_page()
    page.goto(URL)
    page.wait_for_selector(".persona")
    page.evaluate(OVERLAY_JS)
    page.mouse.move(640, 360)
    return ctx, page


def wait_answer_done(page: Page, count: int, timeout: float = 180) -> None:
    """Wait until `count` assistant answers have finished (each finished answer shows a footer)."""
    page.wait_for_function(
        "n => document.querySelectorAll('.msg-assistant .msg-foot').length >= n", arg=count, timeout=timeout * 1000
    )


def record_chat(browser, out_dir: Path) -> dict:
    ctx, page = open_page(browser, out_dir)
    d = Demo(page)
    d.caption("PersonaRAG · chat with AI simulations of public figures, grounded in their own words", hold=3.2)
    d.caption("")

    # 1. Elon: a starter question, streamed answer, then open a citation.
    d.click(".persona", 0)
    d.pause(0.8)
    d.click(".starter", 0)
    d.caption("Retrieves passages from 275 chunks of blogs, interviews and podcasts…")
    page.wait_for_selector(".msg-assistant .answer", timeout=120_000)
    d.caption("…and answers in his documented voice, streaming token by token")
    wait_answer_done(page, 1)
    d.pause(1.2)
    d.caption("Every claim cites a numbered source", hold=1.4)
    d.click(".msg-assistant .cite", 0)
    d.caption("Click a citation to see the exact passage it came from", hold=3.5)
    d.page.mouse.move(1100, 420, steps=20)
    d.page.mouse.wheel(0, 260)
    d.pause(1.6)
    page.keyboard.press("Escape")
    d.caption("")
    d.pause(0.6)

    # 2. Zuckerberg: a typed question.
    d.click(".persona", 1)
    d.pause(0.8)
    d.type(".composer textarea", "How should the US compete with China on AI?")
    page.keyboard.press("Enter")
    d.caption("Hybrid search (BM25 + vectors) + cross-encoder re-ranking picks the evidence")
    wait_answer_done(page, 1)
    d.caption("A grounding badge shows how well the answer is backed by sources", hold=1.0)
    d.move_to(".badge")
    d.pause(2.4)
    d.caption("")

    # 3. Tim Cook: an out-of-scope question; the system should decline rather than invent.
    d.click(".persona", 3)
    d.pause(0.6)
    d.type(".composer textarea", "What's your favourite pizza topping?")
    page.keyboard.press("Enter")
    d.caption("Out-of-scope question: it says so instead of making something up")
    wait_answer_done(page, 1)
    d.pause(3.2)
    d.caption("")
    d.pause(0.5)
    d.mark("end")
    ctx.close()
    return d.marks


def record_panel(browser, out_dir: Path) -> dict:
    ctx, page = open_page(browser, out_dir)
    d = Demo(page)
    d.click(".tab", 1)
    d.caption("Panel mode: several personas answer the same question", hold=1.2)
    d.type(".panel-ask input", "Should I drop out of college to start a company?")
    page.keyboard.press("Enter")
    d.mark("submitted")
    d.caption("Answers stream in as each panelist finishes (sped up)")
    page.wait_for_function("document.querySelectorAll('.panel-card .answer').length >= 1", timeout=600_000)
    d.mark("first_answer")
    page.wait_for_function("document.querySelector('.moderator .answer') !== null", timeout=600_000)
    d.mark("summary")
    d.caption("Then an AI moderator compares where they agree and differ", hold=1.5)
    d.page.mouse.move(700, 500, steps=15)
    for _ in range(6):
        page.mouse.wheel(0, 160)
        d.pause(0.45)
    d.pause(3)
    d.caption("Open source · github.com/sidvw1808/PersonaRAG", hold=3)
    d.mark("end")
    ctx.close()
    return d.marks


def ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


def latest_webm(folder: Path) -> Path:
    return max(folder.glob("*.webm"), key=lambda p: p.stat().st_mtime)


def edit(chat: Path, panel: Path, panel_marks: dict) -> None:
    # Panel: real time until shortly after submitting, 8x through the wait, real time from the moderator on.
    a, b = panel_marks["submitted"] + 1.5, panel_marks["summary"] - 0.5
    filt = (
        f"[1:v]trim=0:{a},setpts=PTS-STARTPTS[p1];"
        f"[1:v]trim={a}:{b},setpts=(PTS-STARTPTS)/8[p2];"
        f"[1:v]trim=start={b},setpts=PTS-STARTPTS[p3];"
        "[0:v]setpts=PTS-STARTPTS[c];"
        "[c][p1][p2][p3]concat=n=4:v=1:a=0,fps=30,format=yuv420p[out]"
    )
    mp4 = OUT / "demo.mp4"
    ffmpeg("-i", str(chat), "-i", str(panel), "-filter_complex", filt, "-map", "[out]",
           "-c:v", "libx264", "-crf", "26", "-preset", "slow", "-movflags", "+faststart", str(mp4))
    # README GIF: the chat segment's first ~32 s at 1.25x, 900 px wide, with an optimised palette.
    gif = OUT / "demo.gif"
    ffmpeg("-i", str(chat), "-t", "32", "-vf",
           "setpts=PTS/1.25,fps=10,scale=900:-1:flags=lanczos,split[s0][s1];"
           "[s0]palettegen=max_colors=96:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
           str(gif))
    for f in (mp4, gif):
        print(f"{f.relative_to(ROOT)}: {f.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (RAW / "chat").mkdir(parents=True, exist_ok=True)
    (RAW / "panel").mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            chat_marks = record_chat(browser, RAW / "chat")
            print("chat segment:", chat_marks)
            panel_marks = record_panel(browser, RAW / "panel")
            print("panel segment:", panel_marks)
        except Exception:
            # Leave a screenshot of whatever page is open, to see which step stalled.
            for ctx in browser.contexts:
                for page in ctx.pages:
                    page.screenshot(path=str(RAW / "failure.png"))
            raise
        finally:
            browser.close()
    (RAW / "marks.json").write_text(json.dumps({"chat": chat_marks, "panel": panel_marks}, indent=2))
    edit(latest_webm(RAW / "chat"), latest_webm(RAW / "panel"), panel_marks)


if __name__ == "__main__":
    main()
