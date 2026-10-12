# Dragon head, built from nothing. +z = snout direction, y up, x = left.
from lab import *


def cranium(p):
    return ellipsoid(p, (0, 0.15, -0.25), (0.30, 0.28, 0.36))


def jowls(p):
    return ellipsoid(mirror_x(p), (0.19, -0.02, -0.15), (0.13, 0.16, 0.22))


def snout(p):
    d = capsule(p, (0, 0.08, 0.0), (0, 0.0, 0.68), 0.21, 0.12)
    return smax(d, p[:, 1] - (0.22 - 0.13 * p[:, 2]), 0.05)  # flat top sloping to the tip


def jaw(p):
    return capsule(p, (0, -0.12, -0.2), (0, -0.16, 0.62), 0.19, 0.08)


def brows(p):
    return ellipsoid(mirror_x(p), (0.17, 0.27, 0.02), (0.10, 0.06, 0.17))


def horns(p):
    q = mirror_x(p)
    pts = [(0.15, 0.33, -0.35), (0.22, 0.47, -0.62), (0.25, 0.50, -0.82), (0.24, 0.43, -0.96)]
    rad = [0.08, 0.055, 0.03, 0.005]
    d = 1e9
    for i in range(3):
        d = smin(d, capsule(q, pts[i], pts[i + 1], rad[i], rad[i + 1]), 0.02)
    return d


def cheek_spikes(p):
    q = mirror_x(p)
    return np.minimum(capsule(q, (0.25, 0.02, -0.28), (0.42, 0.06, -0.58), 0.045, 0.005),
                      capsule(q, (0.24, -0.10, -0.25), (0.38, -0.14, -0.52), 0.035, 0.004))


def neck(p):
    return capsule(p, (0, -0.05, -0.45), (0, -0.55, -0.80), 0.24, 0.26)


def ridge(p):
    d = 1e9
    for t in np.linspace(0, 1, 5):
        base = np.array([0, 0.40 - 0.42 * t, -0.50 - 0.30 * t])
        d = np.minimum(d, capsule(p, base, base + (0, 0.11, -0.07), 0.035, 0.003))
    return d


def mouth_slit(p):
    q = rot(p, "x", -4, center=(0, -0.07, 0.3))
    return box(q, (0, -0.075, 0.37), (0.3, 0.018, 0.40))


def teeth(p):
    q = mirror_x(p)
    d = 1e9
    for t in np.linspace(0, 1, 7):
        x, z = 0.115 - 0.055 * t, 0.10 + 0.50 * t
        d = np.minimum(d, capsule(q, (x, -0.055, z), (x + 0.01, -0.115, z), 0.018, 0.002))
    return d


def eye_sockets(p):
    return sphere(mirror_x(p), (0.21, 0.17, 0.03), 0.065)


def eyes(p):
    return sphere(mirror_x(p), (0.195, 0.165, 0.03), 0.045)


def nostrils(p):
    return sphere(mirror_x(p), (0.07, 0.11, 0.66), 0.03)


def sdf(p):
    d = smin(cranium(p), snout(p), 0.12)
    d = smin(d, jowls(p), 0.08)
    d = smin(d, jaw(p), 0.08)
    d = smin(d, brows(p), 0.06)
    d = smin(d, neck(p), 0.12)
    d = smin(d, horns(p), 0.05)
    d = smin(d, cheek_spikes(p), 0.04)
    d = smin(d, ridge(p), 0.03)
    d = cut(d, mouth_slit(p), 0.015)
    d = np.minimum(d, teeth(p))
    d = cut(d, eye_sockets(p), 0.02)
    d = np.minimum(d, eyes(p))
    d = cut(d, nostrils(p), 0.02)
    return d


PARTS = {"cranium": cranium, "snout": snout, "jaw": jaw, "horns": horns, "neck": neck, "teeth": teeth}
