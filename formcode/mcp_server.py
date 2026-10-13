"""The engine as a stdio MCP server (PROTOCOL §8): one tool, formcode(commands), backed by one Engine per process.
Needs the optional extra: pip install formcode[mcp]. stdout is the protocol channel: the SDK points fd 1 at stderr
while it serves (child processes inherit that), and a call sends sys.stdout to stderr too, else prints buffered in it
(e.g. by code() scenes) would reach the wire when the SDK restores fd 1 at exit."""
import base64
import re
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent

from .session import Engine

DESCRIPTION = """Build a 3D model by commands; the engine checks, versions and reverts for you.
One call = one batch: one Python-style call per line, literal args (never executed). Atomic: any error -> nothing applied.
Meters; y up, +z front, x = subject's left. mirror=True: define at x>0.
at=: "landmark" | rel("mark", dx, dy, dz) | on("host", x=, y=) (ray from front onto host's skin; side="L|R|top|back" + the other 2 coords) | depth=-0.002 (along normal, - = in) | (x, y, z) (counted)
new(name) open("x.fc") save("x.fc") export("x.glb")
add(name, shape, at=, size=, blend=0.01, op="union|cut|intersect|separate", mirror=) shapes: sphere ellipsoid box capsule torus eye ear
mark(name, at)  curve(name, through=[...])  stroke(name, brush, along=, amount=, width=) brushes: crease ridge inflate carve groove
set(name, **params)  move(name, at=)  rename(a, b)  delete(name)
seat(name): engine attaches it. solve("jaw.size", "width jaw", 0.104)
measure(a, b)  measure(name)  report()  tree()  history()  undo()  goto("v004")
look(view="front|side|3/4|back|top|all", at=name): returns an image
Replies list changes and warnings. A regression is REVERTED with "fix n:" lines: fix(n) applies one, keep() accepts anyway."""


def serve(allow_code=False, root=None):
    eng, lock = Engine(allow_code, root), threading.Lock()
    mcp = MCPServer("formcode")

    @mcp.tool(description=DESCRIPTION, structured_output=False)
    def formcode(commands: str) -> CallToolResult:
        with lock, redirect_stdout(sys.stderr):  # sync tools run in worker threads; one engine, one batch at a time
            try:
                reply = eng.do(commands)
            except Exception as e:  # an engine bug: say so instead of the SDK's bare "Error executing tool"
                return CallToolResult(content=[TextContent(type="text", text=f"error: {type(e).__name__}: {e}")], is_error=True)
            dirs = [d for d in (eng.root, eng.tmp) if d]  # only images the engine wrote: inside the root, or its temp dir
            content = [TextContent(type="text", text=reply)]
            for p in re.findall(r"\[image\] (.+?\.png) \(", reply):  # anywhere: results share a line, joined by " | "
                p = Path(p).resolve()
                if p.is_file() and any(p.is_relative_to(Path(d).resolve()) for d in dirs):
                    content.append(ImageContent(type="image", data=base64.b64encode(p.read_bytes()).decode(), mime_type="image/png"))
            return CallToolResult(content=content, is_error=reply.startswith("error line"))

    mcp.run("stdio")
