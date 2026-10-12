"""Client for the Blender worker: JSON lines over the worker process's stdio."""
import json
import os
import re
import shutil
import subprocess
from collections import deque
from glob import glob
from pathlib import Path

WORKER = Path(__file__).resolve().parents[2] / "blender_worker" / "worker.py"
TAG = "@@fc "  # must match blender_worker/worker.py


class WorkerError(RuntimeError):
    pass


def find_blender():
    """FORMCODE_BLENDER env var, else `blender` on PATH, else newest default Windows install."""
    exe = os.environ.get("FORMCODE_BLENDER") or shutil.which("blender")
    if exe:
        return exe
    hits = glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe")
    version = lambda p: [int(n) for n in re.findall(r"\d+", Path(p).parent.name)]
    return max(hits, key=version, default=None)


class Worker:
    def __init__(self, blender=None):
        exe = blender or find_blender()
        if not exe:
            raise WorkerError("Blender not found; set FORMCODE_BLENDER to the blender executable")
        self.proc = subprocess.Popen(
            [exe, "-b", "--factory-startup", "--python", str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        self._id = 0
        self._log = deque(maxlen=20)  # recent non-protocol output, for crash reports

    def call(self, method, **params):
        self._id += 1
        self.proc.stdin.write(json.dumps({"id": self._id, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            if not line.startswith(TAG):
                self._log.append(line.rstrip())
                continue
            resp = json.loads(line[len(TAG):])
            if "error" in resp:
                raise WorkerError(resp["error"])
            return resp["result"]
        raise WorkerError(f"worker exited ({self.proc.wait()}): " + " | ".join(self._log))

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
