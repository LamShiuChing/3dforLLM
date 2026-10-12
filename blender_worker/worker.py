"""Blender-side worker. The only file in the repo that imports bpy.

Run: blender -b --factory-startup --python blender_worker/worker.py
Protocol: one JSON request per stdin line {"id", "method", "params"};
one response per stdout line, prefixed with TAG so Blender's own log noise is skipped.
"""
import json
import sys

import bpy

TAG = "@@fc "


def ping():
    return {"blender": bpy.app.version_string}


def export_cube(path, size=2.0):
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    bpy.ops.mesh.primitive_cube_add(size=size)
    me = bpy.context.object.data
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")
    return {"path": path, "v": len(me.vertices), "f": len(me.polygons)}


METHODS = {"ping": ping, "export_cube": export_cube}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            resp = {"id": rid, "result": METHODS[req["method"]](**req.get("params", {}))}
        except Exception as e:
            resp = {"id": rid, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(TAG + json.dumps(resp) + "\n")
        sys.stdout.flush()


main()
