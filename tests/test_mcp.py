import base64
import subprocess
import sys
from pathlib import Path

import pytest

BALL = 'new("t")\nadd("ball", "sphere", at=(0, 0, 0), size=0.05)'


def test_mcp_server_end_to_end(tmp_path):
    """One stdio server, driven by the SDK's own client: one tool, text replies, isError, images, the warm pool."""
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client, StdioServerParameters

    async def main():
        server = StdioServerParameters(command=sys.executable, args=["-m", "formcode", "mcp", "--root", str(tmp_path)])
        async with Client(server) as c:
            tools = (await c.list_tools()).tools
            assert [t.name for t in tools] == ["formcode"] and "fix(n)" in tools[0].description
            call = lambda s: c.call_tool("formcode", {"commands": s})
            r = await call(BALL)
            assert not r.is_error and r.content[0].text.startswith("ok new t | ok add ball") and len(r.content) == 1
            r = await call('set("bal", size=0.06)')
            assert r.is_error and r.content[0].text.startswith("error line 1") and 'did you mean "ball"' in r.content[0].text
            for view in ("front", "all"):  # one view in threads; "all" in the warm process pool
                r = await call(f'look("{view}")')
                assert not r.is_error and r.content[0].text.startswith("[image] ")
                assert [x.type for x in r.content] == ["text", "image"] and r.content[1].mime_type == "image/png"
                assert base64.b64decode(r.content[1].data)[:8] == b"\x89PNG\r\n\x1a\n"
            r = await call('code("c", "def sdf(p): return p[:, 0]")')
            assert r.is_error and "code execution is off" in r.content[0].text
            r = await call("tree()")
            assert not r.is_error and "ball" in r.content[0].text  # the refused batch left the scene alone

    anyio.run(main)


def test_mcp_stray_prints_stay_off_the_wire(tmp_path, caplog):
    """stdout is the protocol channel: a code() scene that prints must not put a line on it, also at server exit."""
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client, StdioServerParameters

    async def main():
        server = StdioServerParameters(command=sys.executable, args=["-m", "formcode", "mcp", "--allow-code", "--root", str(tmp_path)])
        async with Client(server) as c:
            r = await c.call_tool("formcode", {"commands": '''code("c", "def sdf(p): print('JUNK'); return (p ** 2).sum(1) ** 0.5 - 0.05")'''})
            assert not r.is_error and r.content[0].text.startswith("ok code c")

    anyio.run(main)
    assert "Failed to parse" not in caplog.text


def test_repl_subprocess_multiview_look(tmp_path):
    """python -m formcode repl: a multi-view look spawns the warm render pool; workers must not re-run the CLI."""
    r = subprocess.run([sys.executable, "-m", "formcode", "repl", "--root", str(tmp_path)], input=BALL.replace("\n", "\\\n") + '\nlook("all")\n',
                       capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert out[0].startswith("ok new t | ok add ball") and len(out) == 3, r.stdout
    assert out[2].startswith("[image] ") and Path(out[2].split()[1]).is_file()
    assert "Error" not in r.stderr and "Traceback" not in r.stderr, r.stderr
