"""Live web view: see the turn mechanics in real time at http://localhost:8899.

A thin add-on -- the same session events logs.py narrates are broadcast as
JSON over a local WebSocket, and webview.html renders them: who is speaking,
interim transcripts as you talk, merges, barge-ins, discarded turns, tool
calls. A webview failure must never break the call, so everything is guarded.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import WSMsgType, web
from livekit.agents.voice import io

logger = logging.getLogger("bolt-bites")

PORT = 8899
_HTML = Path(__file__).parent / "webview.html"

_clients: set[web.WebSocketResponse] = set()
_session = None  # set in start(); the mute button needs it
_muted = False
_first_client = False


def _broadcast(payload: dict) -> None:
    payload["at"] = time.time()
    data = json.dumps(payload, ensure_ascii=False)
    for ws in list(_clients):
        try:
            asyncio.create_task(ws.send_str(data))
        except Exception:
            _clients.discard(ws)


async def _index(_request: web.Request) -> web.Response:
    return web.Response(text=_HTML.read_text(), content_type="text/html")


def _set_muted(muted: bool) -> None:
    """Mic gate, held server-side so every tab agrees and reloads can't desync."""
    global _muted
    _muted = muted
    if _session is not None:
        try:
            # Detaches/reattaches the mic stream (the push-to-talk API); safe
            # to call before the audio input exists -- the flag is honored
            # when it attaches.
            _session.input.set_audio_enabled(not muted)
        except Exception:
            logger.exception("webview mute toggle failed (ignored)")
    logger.info("webview: mic %s", "MUTED" if muted else "live")
    _broadcast({"t": "muted", "muted": muted})


async def _socket(request: web.Request) -> web.WebSocketResponse:
    global _first_client
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _clients.add(ws)
    try:
        if not _first_client:
            # The page opens muted (like joining a call) so a demo/recording
            # can be framed before the mic is hot. First visit only: console
            # runs that never open the page keep a live mic, and a mid-call
            # reload doesn't yank the mic away.
            _first_client = True
            _set_muted(True)
        else:
            await ws.send_str(json.dumps({"t": "muted", "muted": _muted, "at": time.time()}))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except ValueError:
                    continue
                if payload.get("t") == "set_muted":
                    _set_muted(bool(payload.get("muted")))
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        _clients.discard(ws)
    return ws


async def start(session) -> None:
    """Serve the page and mirror session events to it. Failure-safe."""
    try:
        app = web.Application()
        app.router.add_get("/", _index)
        app.router.add_get("/ws", _socket)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", PORT).start()
        logger.info("webview: http://localhost:%d (page opens mic-muted)", PORT)
    except Exception:
        logger.exception("webview failed to start (ignored)")
        return

    global _session
    _session = session

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        _broadcast({"t": "user_state", "state": ev.new_state})

    @session.on("agent_state_changed")
    def _agent_state(ev) -> None:
        _broadcast({"t": "agent_state", "state": ev.new_state})
        if ev.old_state == "thinking" and ev.new_state == "listening":
            _broadcast({"t": "discarded"})

    @session.on("user_input_transcribed")
    def _transcribed(ev) -> None:
        _broadcast({"t": "heard" if ev.is_final else "interim", "text": ev.transcript})

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        item = ev.item
        if item.type == "message":
            _broadcast(
                {
                    "t": "msg",
                    "role": item.role,
                    "text": item.text_content,
                    "interrupted": bool(getattr(item, "interrupted", False)),
                }
            )

    @session.on("overlapping_speech")
    def _overlap(ev) -> None:
        _broadcast({"t": "overlap", "interruption": bool(ev.is_interruption)})

    @session.on("agent_false_interruption")
    def _false_interruption(ev) -> None:
        _broadcast({"t": "false_interruption", "resumed": bool(ev.resumed)})

    @session.on("function_tools_executed")
    def _tools(ev) -> None:
        try:
            names = [c.name for c in ev.function_calls]
        except Exception:
            names = []
        _broadcast({"t": "tools", "names": names})


class _TranscriptTee(io.TextOutput):
    """Mirrors the agent's spoken text to the page, then forwards it on.

    Sits in the transcript output chain, where text arrives SYNCED with the
    audio playout -- so the page types her words at the moment you hear them.
    """

    def __init__(self, wrapped: io.TextOutput) -> None:
        super().__init__(label="webview-tee", next_in_chain=wrapped)
        self._wrapped = wrapped

    async def capture_text(self, text: str) -> None:
        _broadcast({"t": "agent_delta", "text": text})
        if self._wrapped:
            await self._wrapped.capture_text(text)

    def flush(self) -> None:
        _broadcast({"t": "agent_flush"})
        if self._wrapped:
            self._wrapped.flush()


def attach_transcript(session) -> None:
    """Install the tee. Call AFTER session.start -- the sink exists then."""
    try:
        session.output.transcription = _TranscriptTee(session.output.transcription)
    except Exception:
        logger.exception("webview transcript tee failed (ignored)")
