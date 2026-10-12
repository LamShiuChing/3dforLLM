# Dragon head, built from nothing. +z = snout direction, y up, x = left.
from lab import *


def cranium(p):
    return ellipsoid(p, (0, 0.15, -0.25), (0.30, 0.28, 0.36))


def snout(p):
    return capsule(p, (0, 0.08, 0.0), (0, 0.0, 0.68), 0.21, 0.12)


def jaw(p):
    return capsule(p, (0, -0.12, -0.2), (0, -0.16, 0.62), 0.19, 0.08)


def brows(p):
    return ellipsoid(mirror_x(p), (0.17, 0.27, 0.02), (0.10, 0.06, 0.17))


def horns(p):
    q = mirror_x(p)
    return smin(capsule(q, (0.15, 0.33, -0.35), (0.24, 0.48, -0.70), 0.08, 0.05),
                capsule(q, (0.24, 0.48, -0.70), (0.26, 0.62, -0.95), 0.05, 0.01), 0.03)


def neck(p):
    return capsule(p, (0, -0.05, -0.45), (0, -0.45, -0.95), 0.24, 0.27)


def eye_sockets(p):
    return sphere(mirror_x(p), (0.21, 0.17, 0.02), 0.07)


def eyes(p):
    return sphere(mirror_x(p), (0.20, 0.16, 0.02), 0.055)


def nostrils(p):
    return sphere(mirror_x(p), (0.07, 0.10, 0.70), 0.035)


def sdf(p):
    d = smin(cranium(p), snout(p), 0.12)
    d = smin(d, jaw(p), 0.08)
    d = smin(d, brows(p), 0.06)
    d = smin(d, neck(p), 0.12)
    d = smin(d, horns(p), 0.05)
    d = cut(d, eye_sockets(p), 0.02)
    d = np.minimum(d, eyes(p))
    d = cut(d, nostrils(p), 0.02)
    return d


PARTS = {"cranium": cranium, "snout": snout, "jaw": jaw, "horns": horns, "neck": neck}
