"""Browser driver: operates the muse.ai web client in headless Chromium over CDP.

Design notes
------------
* One Chromium process; each account gets its own *browser context* (isolated
  cookie jar), so switching accounts never requires clearing cookies.
* One tab per account, reused across requests. Per-account serialisation is
  guaranteed by the account pool (``MUSE2API_ACCOUNT_MAX_CONCURRENCY=1``).
* A follow-up from the same user stays on that tab when the new ``messages``
  continue the conversation already on the page. Anything else opens a fresh
  thread, so two users never share a page.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ...accounts.model import Account
from ...config import Settings
from ...core.prompt import followup_text
from ...errors import (
    UpstreamAuthError,
    UpstreamError,
    UpstreamQuotaError,
    UpstreamRefused,
    UpstreamTimeout,
)
from ...upstream import muse
from ..base import (
    ChatRequest,
    DriverCapabilities,
    ImageRequest,
    InputImage,
    MediaResult,
    MuseDriver,
    SessionInfo,
    VideoRequest,
)
from . import dom
from .cdp import CDPError, CDPSession
from .chromium import ChromiumProcess, find_chromium

log = logging.getLogger(__name__)

POLL_INTERVAL = 0.15
STABLE_POLLS_DONE = 4
STABLE_POLLS_FORCE = 12
# Past this many assistant bubbles the page gets slow and a new thread is cheaper.
HOT_BUBBLE_LIMIT = 16
# How long the page may take to show the prompt it was sent before we call it lost.
SEND_CONFIRM_TIMEOUT = 8.0
# Longest wait for loading skeletons once the page is otherwise ready, in case one
# never goes away.
LOADING_GRACE = 10.0
# Requests that fail on every healthy page too (Sentry tunnel); not worth reporting.
NOISY_REQUESTS = ("/monitoring",)


def is_thread_url(url: str) -> bool:
    """A real muse conversation, not the empty "new thread" page."""
    path = urlsplit(url).path.rstrip("/")
    return path.startswith("/thread/") and path != "/thread/new"


@dataclass
class _Tab:
    account_id: str
    context_id: str
    target_id: str
    session: CDPSession
    hint: str | None = None
    turns: list[tuple[str, str]] = field(default_factory=list)
    image_count: int = 0
    thread_url: str = ""
    quota_note: str = ""
    """Last ignored quota-hint snippet, so a page polled every 0.6s logs it once."""
    busy: bool = False
    stale: bool = False
    """Cookies were renewed; close this tab once it is idle."""
    closed: bool = False
    problems: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    """Recent failed requests and page errors, reported when the page gets stuck."""

    @property
    def has_state(self) -> bool:
        return bool(self.turns or self.thread_url)


class BrowserDriver(MuseDriver):
    name = "browser"
    capabilities = DriverCapabilities(
        chat=True, chat_images=True, image=True, image_edit=True, video=True, renew_session=True
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chromium: ChromiumProcess | None = None
        self._browser: CDPSession | None = None
        # One browser context (cookie jar) per account, holding up to
        # account_max_concurrency tabs; each request checks one tab out.
        self._contexts: dict[str, str] = {}
        self._tabs: dict[str, list[_Tab]] = {}
        self._opening: dict[str, int] = {}
        self._tab_cond = asyncio.Condition()
        self._ctx_lock = asyncio.Lock()

    # ------------------------------------------------------------ lifecycle
    async def startup(self) -> None:
        exe = find_chromium(self.settings.chromium_path)
        self._chromium = ChromiumProcess(
            exe, self.settings.cdp_port, self.settings.profile_dir, self.settings.headless,
            proxy=self.settings.browser_proxy,
        )
        ws_url = await self._chromium.start()
        self._browser = await CDPSession.connect(ws_url)
        log.info("browser driver ready")

    async def shutdown(self) -> None:
        for tabs in list(self._tabs.values()):
            for tab in list(tabs):
                await self._close_tab(tab)
        if self._browser:
            await self._browser.close()
        if self._chromium:
            await self._chromium.stop()

    async def health(self) -> dict[str, Any]:
        ok = self._browser is not None and not self._browser.closed
        tabs = [t for ts in self._tabs.values() for t in ts]
        return {"driver": self.name, "ok": ok, "tabs": len(tabs),
                "busy_tabs": sum(t.busy for t in tabs)}

    # ------------------------------------------------------------ tabs
    async def _checkout(self, account: Account,
                        prefer: Callable[[_Tab], bool] | None = None) -> _Tab:
        """Reserve an idle tab of ``account``, opening a new one while under the
        per-account limit. An idle tab matching ``prefer`` wins; without one, a new
        tab beats reusing (and so wiping) a tab that holds another conversation.
        Pair every call with ``_checkin``."""
        limit = max(1, self.settings.account_max_concurrency)
        dead: list[_Tab] = []
        async with self._tab_cond:
            while True:
                tabs = self._tabs.setdefault(account.id, [])
                for t in [t for t in tabs
                          if t.closed or (not t.busy and (t.stale or t.session.closed))]:
                    tabs.remove(t)
                    dead.append(t)
                idle = [t for t in tabs if not t.busy]
                can_open = len(tabs) + self._opening.get(account.id, 0) < limit
                match = next((t for t in idle if prefer is None or prefer(t)), None)
                if match or (idle and not can_open):
                    tab = match or next((t for t in idle if not t.has_state), idle[0])
                    tab.busy = True
                    break
                if can_open:
                    self._opening[account.id] = self._opening.get(account.id, 0) + 1
                    tab = None
                    break
                # The pool caps requests per account, so this only waits out a tab
                # that is still being closed.
                await self._tab_cond.wait()
        for t in dead:
            await self._close_tab(t)
        if tab:
            return tab
        try:
            tab = await self._open_tab(account)
        finally:
            async with self._tab_cond:
                self._opening[account.id] -= 1
                if tab:
                    tab.busy = True
                    self._tabs.setdefault(account.id, []).append(tab)
                self._tab_cond.notify_all()
        log.info("opened tab %d for account %s", len(self._tabs[account.id]), account.id)
        return tab

    async def _checkin(self, tab: _Tab) -> None:
        # Free the tab before taking the lock: a request cancelled while waiting for
        # the lock (client gone) must not leave its tab marked busy for good.
        tab.busy = False
        await asyncio.shield(self._notify_tabs())
        if tab.stale:
            await self._close_tab(tab)

    async def _notify_tabs(self) -> None:
        async with self._tab_cond:
            self._tab_cond.notify_all()

    async def _context(self, account: Account) -> str:
        async with self._ctx_lock:
            if ctx := self._contexts.get(account.id):
                return ctx
            assert self._browser
            res = await self._browser.send("Target.createBrowserContext", {"disposeOnDetach": False})
            self._contexts[account.id] = res["browserContextId"]
            return res["browserContextId"]

    async def _open_tab(self, account: Account) -> _Tab:
        if missing := muse.missing_session_cookies(account.cookies):
            raise UpstreamAuthError(f"account is missing session cookies: {', '.join(missing)}")
        assert self._browser and self._chromium
        context_id = await self._context(account)
        # A window per tab: Chromium throttles background tabs of a shared window,
        # which made parallel requests on one account run one after another.
        target = await self._browser.send(
            "Target.createTarget",
            {"url": "about:blank", "browserContextId": context_id, "newWindow": True},
        )
        target_id = target["targetId"]
        session = await CDPSession.connect(
            f"ws://127.0.0.1:{self._chromium.port}/devtools/page/{target_id}"
        )
        for domain in ("Page", "Runtime", "Network"):
            await session.send(f"{domain}.enable")
        await session.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
        await session.send("Page.setWebLifecycleState", {"state": "active"})
        await self._set_cookies(session, account)
        tab = _Tab(account.id, context_id, target_id, session)
        self._watch_page(tab)
        return tab

    @staticmethod
    def _watch_page(tab: _Tab) -> None:
        """Record failed requests and page errors so a stuck page can say why."""
        pending: dict[str, str] = {}

        def note(what: str) -> None:
            tab.problems.append(f"{time.strftime('%H:%M:%S')} {what}")

        def sent(p: dict) -> None:
            req = p.get("request") or {}
            url = (req.get("url") or "").split("?")[0]
            if url.startswith("http") and "/_next/static/" not in url:
                pending[p["requestId"]] = f"{req.get('method', 'GET')} {url}"

        def responded(p: dict) -> None:
            res = p.get("response") or {}
            what = pending.get(p.get("requestId", ""))
            if what and (res.get("status") or 0) >= 400 and not any(n in what for n in NOISY_REQUESTS):
                note(f"HTTP {res['status']} {what}")

        def failed(p: dict) -> None:
            what = pending.pop(p.get("requestId", ""), None)
            if what and not p.get("canceled") and not any(n in what for n in NOISY_REQUESTS):
                note(f"{p.get('blockedReason') or p.get('errorText') or 'failed'} {what}")

        def thrown(p: dict) -> None:
            d = p.get("exceptionDetails") or {}
            note("exception " + ((d.get("exception") or {}).get("description") or d.get("text") or "")[:200])

        def console(p: dict) -> None:
            if p.get("type") == "error":
                args = p.get("args") or [{}]
                note("console.error " + str(args[0].get("value") or args[0].get("description") or "")[:200])

        tab.session.on("Network.requestWillBeSent", sent)
        tab.session.on("Network.responseReceived", responded)
        tab.session.on("Network.loadingFinished", lambda p: pending.pop(p.get("requestId", ""), None))
        tab.session.on("Network.loadingFailed", failed)
        tab.session.on("Runtime.exceptionThrown", thrown)
        tab.session.on("Runtime.consoleAPICalled", console)

    async def _close_tab(self, tab: _Tab) -> None:
        """Idempotent; marks the tab closed first so a cancelled close never leaves it
        counted as open or busy."""
        if tab.closed:
            return
        tab.closed, tab.busy = True, False
        await asyncio.shield(self._dispose_tab(tab))

    async def _dispose_tab(self, tab: _Tab) -> None:
        async with self._tab_cond:
            tabs = self._tabs.get(tab.account_id, [])
            if tab in tabs:
                tabs.remove(tab)
            last = not tabs and not self._opening.get(tab.account_id)
            context_id = self._contexts.pop(tab.account_id, None) if last else None
            self._tab_cond.notify_all()
        await tab.session.close()
        if self._browser and not self._browser.closed:
            with contextlib.suppress(Exception):
                await self._browser.send("Target.closeTarget", {"targetId": tab.target_id})
            if context_id:
                with contextlib.suppress(Exception):
                    await self._browser.send(
                        "Target.disposeBrowserContext", {"browserContextId": context_id}
                    )

    @staticmethod
    async def _set_cookies(session: CDPSession, account: Account) -> None:
        floor = time.time() + 3600
        for name, value in account.cookies.items():
            if not value:
                continue
            exp = account.cookie_expires.get(name) or 0
            params = {
                "name": name,
                "value": value,
                "domain": ".muse.ai",
                "path": "/",
                "secure": True,
                # Never hand Chromium a past expiry; it would drop the cookie silently.
                "expires": exp if exp > floor else time.time() + 7 * 86400,
            }
            await session.send("Network.setCookie", params)

    @staticmethod
    def _forget(tab: _Tab) -> None:
        tab.hint = None
        tab.turns = []
        tab.image_count = 0
        tab.thread_url = ""

    def _load_hot(self, tab: _Tab, account: Account) -> None:
        """Restore the last conversation after a restart or a new browser tab."""
        if tab.turns or tab.thread_url:
            return
        hot = account.meta.get("hot_page")
        if not isinstance(hot, dict):
            return
        turns = []
        for item in hot.get("turns") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                turns.append((str(item[0]), str(item[1])))
        tab.turns = turns
        tab.hint = hot.get("hint") or None
        tab.thread_url = str(hot.get("url") or "")
        try:
            tab.image_count = int(hot.get("image_count") or 0)
        except (TypeError, ValueError):
            tab.image_count = 0

    def _save_hot(self, tab: _Tab, account: Account, req: ChatRequest, url: str) -> None:
        if url:
            tab.thread_url = url
        tab.hint = req.conversation_hint
        tab.turns = list(req.turns)
        tab.image_count = len(req.images)
        account.meta["hot_page"] = {
            "hint": req.conversation_hint or "",
            "turns": [list(turn) for turn in req.turns],
            "url": tab.thread_url,
            "image_count": tab.image_count,
        }

    async def _href(self, tab: _Tab) -> str:
        try:
            return (await tab.session.evaluate("location.href")) or ""
        except CDPError:
            return ""

    async def _on_thread(self, tab: _Tab) -> bool:
        """The open page is the saved conversation and can take another message."""
        href = await self._href(tab)
        if not is_thread_url(href):
            return False
        if tab.thread_url and urlsplit(href).path.rstrip("/") != urlsplit(tab.thread_url).path.rstrip("/"):
            return False
        try:
            page = await tab.session.evaluate(dom.PAGE_STATE) or {}
            if not page.get("hasInput"):
                return False
            if "Connecting..." in (page.get("head") or ""):
                return False
            state = await self._state(tab)
        except CDPError:
            return False
        if state.get("generating"):
            return False
        if any(hint in state.get("tail", "") for hint in dom.STALL_HINTS):
            return False
        return (state.get("agentCount") or 0) < HOT_BUBBLE_LIMIT

    async def _new_thread(self, tab: _Tab) -> None:
        self._forget(tab)
        tab.problems.clear()
        await tab.session.send("Page.navigate", {"url": muse.NEW_THREAD_URL})
        deadline = time.monotonic() + self.settings.page_ready_timeout
        state: dict = {}
        unanswered = 0
        ready_since: float | None = None
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            # A busy page can leave Runtime.evaluate unanswered while it loads; poll again.
            try:
                state = await tab.session.evaluate(dom.PAGE_STATE, timeout=5.0) or {}
            except (CDPError, TimeoutError):
                unanswered += 1
                continue
            if not state.get("ready"):
                ready_since = None
                continue
            ready_since = ready_since or time.monotonic()
            if not state.get("loading"):
                return
            if time.monotonic() - ready_since > LOADING_GRACE:
                log.info("muse.ai still shows %s; using the page anyway", state["loading"])
                return
        head = (state.get("head") or "").lower()
        if any(h in head for h in dom.LOGIN_HINTS):
            raise UpstreamAuthError("muse.ai redirected to login; cookies are no longer valid")
        reason = self._not_ready_reason(state, unanswered)
        log.warning("new thread not ready after %.0fs (%s): path=%s hasInput=%s hydration=%s "
                    "unanswered=%d head=%r problems=%s", self.settings.page_ready_timeout, reason,
                    state.get("path"), state.get("hasInput"), state.get("hydration"),
                    unanswered, " ".join((state.get("head") or "").split())[:300],
                    list(tab.problems))
        raise UpstreamTimeout(f"muse.ai page did not become ready: {reason}")

    @staticmethod
    def _not_ready_reason(state: dict, unanswered: int) -> str:
        if not state:
            return f"page never answered ({unanswered} evaluations timed out)"
        if "Connecting..." in (state.get("head") or ""):
            return "workspace stuck on 'Connecting...'"
        if not state.get("hasInput"):
            return f"no chat input on {state.get('path') or 'page'}"
        if state.get("hydration") not in (None, "hydrated"):
            return f"page not hydrated ({state.get('hydration')})"
        return "page still loading"

    # ------------------------------------------------------------ input
    async def _attach(self, tab: _Tab, images: list[InputImage]) -> None:
        for idx, img in enumerate(images):
            ext = img.mime.split("/")[-1].replace("jpeg", "jpg")
            res = await tab.session.evaluate(
                dom.attach_file(base64.b64encode(img.data).decode(), img.mime, f"input_{idx}.{ext}")
            )
            if not (res or {}).get("ok"):
                raise UpstreamError(f"failed to attach image: {(res or {}).get('err')}")
        if images:
            await asyncio.sleep(1.0)

    @staticmethod
    async def _click(tab: _Tab, x: float, y: float) -> None:
        await tab.session.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        for kind in ("mousePressed", "mouseReleased"):
            await tab.session.send("Input.dispatchMouseEvent",
                                   {"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1})

    async def _send(self, tab: _Tab, prompt: str) -> None:
        before = await tab.session.evaluate(dom.SEND_STATE) or {}
        for attempt in (1, 2):
            how, button = await self._submit(tab, prompt)
            after = await self._await_posted(tab, before)
            if after is None:
                return
            # An emptied composer with nothing posted means the page dropped the prompt
            # (it was not quite ready); nothing went out, so one more try is safe.
            if attempt == 1 and not after.get("input"):
                log.info("prompt dropped by the page (sent by %s), sending again", how)
                await asyncio.sleep(1.0)
                continue
            break
        input_state = "kept" if after.get("input") else "cleared"
        log.warning("prompt not accepted: sent by %s (send button %s), input %s, before=%s after=%s "
                    "problems=%s", how, button, input_state, before, after, list(tab.problems))
        raise UpstreamError(f"muse.ai did not accept the prompt (sent by {how}, send button "
                            f"{button}, input {input_state})")

    async def _submit(self, tab: _Tab, prompt: str) -> tuple[str, str]:
        """Type the prompt and press Send; returns how it was sent and the button state."""
        res = await tab.session.evaluate(dom.FOCUS_INPUT)
        if not (res or {}).get("ok"):
            raise UpstreamError(f"cannot focus chat input: {(res or {}).get('err', 'not focused')}")
        await tab.session.send("Input.insertText", {"text": prompt})
        button = "missing"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            btn = await tab.session.evaluate(dom.SEND_BUTTON) or {}
            button = btn.get("state", "missing")
            if button == "ready":
                await self._click(tab, btn["x"], btn["y"])
                return "click", "clicked"
            await asyncio.sleep(0.1)
        for kind in ("keyDown", "keyUp"):
            await tab.session.send(
                "Input.dispatchKeyEvent",
                {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13,
                 **({"text": "\r"} if kind == "keyDown" else {})},
            )
        return "enter", button

    @staticmethod
    async def _await_posted(tab: _Tab, before: dict) -> dict | None:
        """None once the page shows the prompt went out, else the last SEND_STATE."""
        after: dict = {}
        deadline = time.monotonic() + SEND_CONFIRM_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            after = await tab.session.evaluate(dom.SEND_STATE) or {}
            if (after.get("users", 0) > before.get("users", 0) or after.get("generating")
                    or after.get("path") != before.get("path")):
                return None
        return after

    async def _state(self, tab: _Tab) -> dict:
        state = await tab.session.evaluate(dom.CHAT_STATE) or {}
        tail = state.get("tail", "")
        low = tail.lower()
        if hit := next((h for h in dom.QUOTA_HINTS if h in low), None):
            i = low.find(hit)
            snippet = " ".join(tail[max(0, i - 120):i + 120].split())
            # The page tail also holds the prompt and the thread-history sidebar, and
            # matching it cooled down accounts whose images muse.ai went on to deliver.
            # Only a finished agent reply that says so counts; anything else is logged.
            reply = state.get("lastText", "")
            if not state.get("generating") and hit in reply.lower():
                log.warning("quota hint %r in reply on %s: %s", hit,
                            tab.thread_url or "new thread", snippet)
                raise UpstreamQuotaError(f"muse.ai reports the account is out of quota: …{snippet}…")
            if snippet != tab.quota_note:
                tab.quota_note = snippet
                log.info("ignoring quota hint %r outside the reply on %s: %s", hit,
                         tab.thread_url or "new thread", snippet)
        return state

    async def _prepare(self, tab: _Tab, prompt: str, images: list[InputImage]) -> dict:
        await self._new_thread(tab)
        base = await self._state(tab)
        await self._attach(tab, images)
        await self._send(tab, prompt)
        return base

    async def _deny_approvals(self, tab: _Tab) -> None:
        denied = await tab.session.evaluate(dom.DENY_APPROVALS) or []
        for what in denied:
            log.info("denied muse.ai %s", what.lower())

    # ------------------------------------------------------------ chat
    async def _live_continuation(self, tab: _Tab) -> bool:
        """This tab already shows the stored conversation, even if its URL is still /thread/new."""
        last_user = next((text for role, text in reversed(tab.turns) if role == "user"), "")
        if not last_user:
            return False
        try:
            page = await tab.session.evaluate(dom.PAGE_STATE) or {}
            if not page.get("hasInput"):
                return False
            state = await self._state(tab)
            present = await tab.session.evaluate(
                f"!!(document.body && document.body.innerText.includes({dom.q(last_user[:80])}))"
            )
        except CDPError:
            return False
        if state.get("generating"):
            return False
        if any(hint in state.get("tail", "") for hint in dom.STALL_HINTS):
            return False
        if (state.get("agentCount") or 0) >= HOT_BUBBLE_LIMIT:
            return False
        return bool(present)

    async def _open_saved(self, tab: _Tab) -> None:
        await tab.session.send("Page.navigate", {"url": tab.thread_url})
        deadline = time.monotonic() + self.settings.page_ready_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            if await self._on_thread(tab):
                return
        raise UpstreamTimeout("saved conversation did not become ready")

    async def _begin_chat(self, account: Account, req: ChatRequest):
        """Send ``req`` on the saved conversation when it continues that page."""
        hint = req.conversation_hint or ""
        tab = await self._checkout(account, prefer=lambda t: t.has_state and (t.hint or "") == hint)
        prompt, images = req.prompt, req.images
        try:
            # The saved conversation (restored after a restart) may only go into a
            # tab while no other tab of the account already holds a conversation.
            if not any(t.has_state for t in self._tabs.get(account.id, []) if t is not tab):
                self._load_hot(tab, account)
            same_owner = (tab.hint or "") == (req.conversation_hint or "")
            follow = followup_text(tab.turns, req.turns) if tab.turns and same_owner else None
            if follow and len(req.images) >= tab.image_count and (
                tab.thread_url or await self._live_continuation(tab)
            ):
                if await self._on_thread(tab) or await self._live_continuation(tab):
                    log.info("reusing hot page %s", tab.thread_url or "open page")
                elif tab.thread_url:
                    log.info("reopening %s", tab.thread_url)
                    await self._open_saved(tab)
                prompt = follow
                images = req.images[tab.image_count:]
            else:
                log.info("opening a new thread")
                saved = account.meta.get("hot_page")
                if isinstance(saved, dict) and (saved.get("hint") or "") == hint:
                    account.meta.pop("hot_page", None)
                await self._new_thread(tab)
            base = await self._state(tab)
            await self._attach(tab, images)
            await self._send(tab, prompt)
            return tab, base
        except CDPError as exc:
            await self._close_tab(tab)
            await self._checkin(tab)
            raise UpstreamError(f"browser error: {exc}") from exc
        except BaseException:
            self._forget(tab)
            await self._checkin(tab)
            raise

    async def chat_stream(self, account: Account, req: ChatRequest) -> AsyncIterator[str]:
        tab, base = await self._begin_chat(account, req)
        base_count, base_text = base.get("agentCount", 0), base.get("lastText", "")
        started = time.monotonic()
        emitted, last, stable, got_first = "", None, 0, False
        finished = False
        st: dict = {}
        try:
            while True:
                if req.cancel and req.cancel.is_set():
                    return
                elapsed = time.monotonic() - started
                if elapsed > req.timeout:
                    raise UpstreamTimeout("assistant reply timed out")
                if not got_first and elapsed > req.first_token_timeout:
                    unsent = not await tab.session.evaluate(dom.INPUT_EMPTY)
                    log.warning("no first token after %.0fs: path=%s unsent=%s generating=%s "
                                "agentCount=%s (base %s) lastText=%r tail=%r problems=%s",
                                elapsed, await self._href(tab), unsent, st.get("generating"),
                                st.get("agentCount"), base_count, (st.get("lastText") or "")[:120],
                                " ".join((st.get("tail") or "").split())[-300:], list(tab.problems))
                    raise UpstreamTimeout("no first token from assistant"
                                          + (": the prompt was never sent" if unsent else ""))
                await asyncio.sleep(POLL_INTERVAL)
                st = await self._state(tab)
                if st.get("approval"):
                    await self._deny_approvals(tab)
                    continue
                text = st.get("lastText", "")
                is_new = st.get("agentCount", 0) > base_count or (text and text != base_text)
                if not is_new or not text:
                    if elapsed > 15 and any(h in st.get("tail", "") for h in dom.STALL_HINTS):
                        raise UpstreamTimeout("upstream workspace is stuck connecting")
                    continue
                got_first = True
                if text != last:
                    delta = text[len(emitted):] if text.startswith(emitted) else text
                    if delta:
                        emitted = text
                        yield delta
                    last, stable = text, 0
                else:
                    stable += 1
                    if (not st.get("generating") and stable >= STABLE_POLLS_DONE) or stable >= STABLE_POLLS_FORCE:
                        finished = True
                        return
        except CDPError as exc:
            await self._close_tab(tab)
            raise UpstreamError(f"browser error: {exc}") from exc
        finally:
            if finished:
                url = ""
                for _ in range(10):
                    url = await self._href(tab)
                    if is_thread_url(url):
                        break
                    await asyncio.sleep(0.2)
                self._save_hot(tab, account, req, url if is_thread_url(url) else "")
                if tab.thread_url:
                    log.info("saved hot page %s", tab.thread_url)
                else:
                    log.info("reply finished but the thread url was not ready yet")
            else:
                self._forget(tab)
            await self._checkin(tab)

    # ------------------------------------------------------------ media
    @staticmethod
    def _media_prompt(prompt: str, kind: str, size: str | None, duration: int | None = None) -> str:
        hints = []
        if size:
            hints.append(f"aspect ratio {size}")
        if duration:
            hints.append(f"{duration} seconds long")
        verb = "Generate an image" if kind == "image" else "Generate a video"
        suffix = f" ({', '.join(hints)})" if hints else ""
        return f"{verb}{suffix}: {prompt}"

    # muse.ai posts a completion sentence ("Here's your video…") a few seconds
    # before the media attachment's blob src is ready, so a reply that is "text
    # only" right now may still be finalising. Keep waiting this long for the
    # attachment to appear/load before treating the reply as a refusal.
    _MEDIA_GRACE = {"video": 90.0, "image": 8.0}
    # The timeout is soft while muse.ai is visibly still working (stop button
    # shown or attachment loading): giving up then threw away images that landed
    # a minute later, and the failover started the whole generation over.
    _OVERRUN_FACTOR = 2.5

    async def _wait_media(self, tab: _Tab, base: dict, kind: str, timeout: float,
                          on_progress, cancel: asyncio.Event | None) -> dict:
        base_atts = len(base.get("attachments", []))
        base_count = base.get("agentCount", 0)
        grace = self._MEDIA_GRACE.get(kind, 8.0)
        started = time.monotonic()
        text_done_at: float | None = None
        denied = asked_to_attach = busy = False
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= timeout and not (busy and elapsed < timeout * self._OVERRUN_FACTOR):
                break
            if cancel and cancel.is_set():
                raise asyncio.CancelledError
            await asyncio.sleep(0.6)
            st = await self._state(tab)
            if st.get("approval"):
                await self._deny_approvals(tab)
                denied, text_done_at = True, None
                continue
            new_atts = st.get("attachments", [])[base_atts:]
            fresh = [a for a in new_atts if a.get("src") and a.get("kind") == kind]
            if fresh:
                return fresh[-1]
            busy = bool(st.get("generating")) or any(kind in (a.get("tid") or "") for a in new_atts)
            elapsed = time.monotonic() - started
            if on_progress:
                on_progress(min(95, int(elapsed / timeout * 100)))
            # An attachment node for this kind is on the page but its (blob) src
            # has not loaded yet -> media is still finalising, keep waiting.
            if any(kind in (a.get("tid") or "") for a in new_atts):
                text_done_at = None
                continue
            # No attachment yet. The reply may be genuinely text-only (a refusal),
            # or the attachment may simply lag the completion sentence.
            text = st.get("lastText", "")
            if st.get("agentCount", 0) > base_count and text and not st.get("generating"):
                if text_done_at is None:
                    text_done_at = time.monotonic()
                elif time.monotonic() - text_done_at >= grace:
                    if not asked_to_attach:
                        # The agent often finishes with the file only in its workspace:
                        # after a denied upload, after post-processing that it shows as
                        # an image gallery (raw, QC crops and final mixed together), or
                        # when it recognises a repeat request and points at the earlier
                        # file. Asking once gets the final file attached in chat.
                        log.info("%s not attached (%s); asking for it in chat: %.120s", kind,
                                 "denied approval" if denied else "text-only reply", text)
                        await self._send(tab, f"Please attach the final {kind} file here in "
                                              "the chat" + (" instead of uploading it." if denied
                                                            else ", as a single attachment."))
                        asked_to_attach, text_done_at = True, None
                        base_count = st.get("agentCount", 0)
                        continue
                    # Last resort: media delivered as a link in the text bubble.
                    probe = await tab.session.evaluate(dom.LAST_BUBBLE_MEDIA) or {}
                    url = self._pick_media_url(probe.get("links", []), kind)
                    if url:
                        log.info("%s delivered as link in text bubble: %s", kind, url)
                        return {"src": url, "kind": kind}
                    log.info("%s: no media after %.0fs grace; links=%s html=%.300s",
                             kind, grace, probe.get("links", []), probe.get("html", ""))
                    raise UpstreamRefused(f"upstream replied with text only: {text[:200]}")
            else:
                text_done_at = None
        if busy:
            log.warning("%s still generating after %.0fs; giving up", kind, elapsed)
        raise UpstreamTimeout(f"{kind} generation timed out")

    _VIDEO_EXT = (".mp4", ".webm", ".mov", ".m4v")
    _IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif")
    _MEDIA_HOSTS = ("videodelivery.net", "cloudflarestream.com", "cloudflarestorage.com",
                    "amazonaws.com", "storage.googleapis.com", "blob.core.windows.net",
                    "musecdn", "cdn.muse", "muse-cdn")
    _MEDIA_PATHS = ("/media/", "/download", "/dl/", "/video", "/attachment", "/files/")

    @classmethod
    def _pick_media_url(cls, links: list[str], kind: str) -> str | None:
        """Best media URL from a reply bubble, or None if nothing looks like one."""
        exts = cls._VIDEO_EXT if kind == "video" else cls._IMAGE_EXT

        def score(u: str) -> int:
            low = u.lower()
            path = low.split("?", 1)[0].split("#", 1)[0]
            if u.startswith("blob:") or path.endswith(exts):
                return 3
            if any(h in low for h in cls._MEDIA_HOSTS):
                return 2
            if any(seg in low for seg in cls._MEDIA_PATHS):
                return 1
            return 0

        cands = [u for u in links if u and not u.startswith(("mailto:", "javascript:"))]
        cands.sort(key=score, reverse=True)
        return cands[0] if cands and score(cands[0]) > 0 else None

    async def _download(self, tab: _Tab, att: dict, kind: str) -> MediaResult:
        res = await tab.session.evaluate(dom.fetch_as_base64(att["src"]), await_promise=True,
                                         timeout=300)
        if not (res or {}).get("ok"):
            raise UpstreamError(f"failed to download generated {kind}: {(res or {}).get('err')}")
        mime = res.get("mime") or ("video/mp4" if kind == "video" else "image/png")
        return MediaResult(data=base64.b64decode(res["b64"]), mime=mime, kind=kind,
                           width=att.get("w") or None, height=att.get("h") or None)

    async def _generate(self, account: Account, prompt: str, images: list[InputImage],
                        kind: str, timeout: float, on_progress, cancel) -> MediaResult:
        # Media always starts a fresh thread, so keep tabs holding a chat for chats.
        tab = await self._checkout(account, prefer=lambda t: not t.has_state)
        try:
            base = await self._prepare(tab, prompt, images)
            att = await self._wait_media(tab, base, kind, timeout, on_progress, cancel)
            return await self._download(tab, att, kind)
        except CDPError as exc:
            await self._close_tab(tab)
            raise UpstreamError(f"browser error: {exc}") from exc
        finally:
            await self._checkin(tab)

    async def generate_image(self, account: Account, req: ImageRequest) -> list[MediaResult]:
        prompt = self._media_prompt(req.prompt, "image", req.size)
        results = []
        for _ in range(max(1, req.n)):
            r = await self._generate(account, prompt, req.reference_images, "image",
                                     req.timeout, req.on_progress, req.cancel)
            r.revised_prompt = req.prompt
            results.append(r)
        return results

    async def generate_video(self, account: Account, req: VideoRequest) -> MediaResult:
        prompt = self._media_prompt(req.prompt, "video", req.size, req.duration)
        images = [req.first_frame] if req.first_frame else []
        return await self._generate(account, prompt, images, "video", req.timeout,
                                    req.on_progress, req.cancel)

    # ------------------------------------------------------------ session
    async def renew_session(self, account: Account) -> SessionInfo:
        info = await muse.renew_session(account.cookies)
        # Retire the account's tabs so new ones load the rotated cookies; a tab
        # in the middle of a request is closed when it is checked back in.
        async with self._tab_cond:
            tabs = list(self._tabs.get(account.id, []))
            for tab in tabs:
                tab.stale = True
            idle = [t for t in tabs if not t.busy]
        for tab in idle:
            await self._close_tab(tab)
        return info
