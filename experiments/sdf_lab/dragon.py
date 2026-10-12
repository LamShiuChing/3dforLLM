# Dragon head, built from nothing. +z = snout direction, y up, x = left.
from lab import *

# neck as an S-curve: two segments, radii at joints. Spikes are placed relative to it.
NECK = [((0, 0.0, -0.40), 0.24), ((0, -0.30, -0.66), 0.25), ((0, -0.66, -0.70), 0.27)]


def cranium(p):
    return ellipsoid(p, (0, 0.15, -0.25), (0.30, 0.28, 0.36))


def jowls(p):
    return ellipsoid(mirror_x(p), (0.19, -0.02, -0.15), (0.13, 0.16, 0.22))


def snout(p):
    d = capsule(p, (0, 0.08, 0.0), (0, 0.02, 0.66), 0.21, 0.11)
    d = smax(d, p[:, 1] - (0.22 - 0.13 * p[:, 2]), 0.05)  # flat top sloping to the tip
    d = smax(d, p[:, 2] - 0.75, 0.04)  # blunt front
    nostril_flare = ellipsoid(mirror_x(p), (0.075, 0.115, 0.63), (0.05, 0.035, 0.06))
    return smin(d, nostril_flare, 0.03)


def jaw(p):
    return capsule(p, (0, -0.12, -0.2), (0, -0.15, 0.6), 0.17, 0.07)  # narrower: upper jaw overhangs


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
    d = 1e9
    for (a, ra), (b, rb) in zip(NECK, NECK[1:]):
        d = smin(d, capsule(p, a, b, ra, rb), 0.08)
    return d


def ridge(p):
    """Dorsal spikes rooted on the neck surface, sweeping backward, shrinking toward the body."""
    d, k = 1e9, 0
    for (a, ra), (b, rb) in zip(NECK, NECK[1:]):
        a, b = np.array(a), np.array(b)
        axis = (b - a) / np.linalg.norm(b - a)
        up = np.array([0, -axis[2], axis[1]])  # perpendicular, dorsal side
        up = up if up[1] > 0 or up[2] < 0 else -up
        for t in (0.15, 0.5, 0.85):
            r = ra + (rb - ra) * t
            base = a + (b - a) * t + up * (r - 0.02)
            h = 0.13 - 0.015 * k
            d = np.minimum(d, capsule(p, base, base + up * h + axis * h * 0.5, 0.04, 0.003))
            k += 1
    for z, h in ((-0.18, 0.06), (-0.33, 0.08)):  # two small ones on the skull
        base = np.array([0, 0.40 - 0.1 * (z + 0.18), z])
        d = np.minimum(d, capsule(p, base, base + (0, h, -h * 0.6), 0.03, 0.003))
    return d


def chin(p):
    return ellipsoid(p, (0, -0.25, 0.28), (0.08, 0.05, 0.22))


def throat_plates(p):
    """Ventral scutes: horizontal bands along the front of the neck, clipped to the neck."""
    d = 1e9
    for (a, ra), (b, rb) in zip(NECK, NECK[1:]):
        a, b = np.array(a), np.array(b)
        axis = (b - a) / np.linalg.norm(b - a)
        belly = -np.array([0, -axis[2], axis[1]])
        belly = belly if belly[1] < 0 or belly[2] > 0 else -belly
        for t in np.linspace(0.04, 0.96, 8):
            c = a + (b - a) * t + belly * ((ra + (rb - ra) * t) - 0.03)
            d = np.minimum(d, capsule(p, c - (0.17, 0, 0), c + (0.17, 0, 0), 0.022))
    return smax(d, neck(p) - 0.025, 0.01)


def scales(q):
    """Two scale sizes: fine on the face/jaw, coarse on the crown and back of the neck."""
    groove = lambda cell, seed: 1 - np.clip((lambda f: (f[1] - f[0]) / (0.28 * cell))(voronoi(q, cell, seed)), 0, 1)
    coarse = np.clip((q[:, 1] - 0.15) / 0.15 + (-0.35 - q[:, 2]) / 0.2, 0, 1)  # up high or far back
    return 0.005 * ((1 - coarse) * groove(0.026, 3) + coarse * groove(0.055, 4))


def mouth_slit(p):
    q = rot(p, "x", -4, center=(0, -0.07, 0.3))
    return box(q, (0, -0.075, 0.37), (0.3, 0.013, 0.40))


def teeth(p):
    q = mirror_x(p)
    d = 1e9
    for i, t in enumerate(np.linspace(0, 1, 6)):
        x, z = 0.10 - 0.05 * t, 0.12 + 0.48 * t
        L = 0.15 if i == 4 else 0.085 - 0.02 * t  # one fang near the front
        d = np.minimum(d, capsule(q, (x + 0.01, -0.06, z), (x + 0.02, -0.06 - L, z + 0.01), 0.02 if i == 4 else 0.014, 0.002))
    return d


def eye_sockets(p):
    return sphere(mirror_x(p), (0.21, 0.17, 0.03), 0.065)


def eyes(p):
    return sphere(mirror_x(p), (0.195, 0.165, 0.035), 0.042)


def eyelids(p):
    q = rot(mirror_x(p), "z", -15, center=(0.2, 0.2, 0.035))  # angled down toward the snout: a scowl
    return ellipsoid(q, (0.205, 0.198, 0.04), (0.065, 0.03, 0.065))


def nostrils(p):
    return sphere(mirror_x(p), (0.075, 0.13, 0.665), 0.025)


def sdf(p):
    d = smin(cranium(p), snout(p), 0.12)
    d = smin(d, jowls(p), 0.08)
    d = smin(d, jaw(p), 0.08)
    d = smin(d, brows(p), 0.06)
    d = smin(d, neck(p), 0.12)
    d = smin(d, horns(p), 0.05)
    d = smin(d, cheek_spikes(p), 0.04)
    d = smin(d, ridge(p), 0.03)
    d = smin(d, chin(p), 0.05)
    d = near_surface(d, p, scales)
    d = smin(d, throat_plates(p), 0.01)
    d = cut(d, mouth_slit(p), 0.01)
    d = np.minimum(d, teeth(p))
    d = cut(d, eye_sockets(p), 0.02)
    d = np.minimum(d, eyes(p))
    d = smin(d, eyelids(p), 0.01)
    d = cut(d, nostrils(p), 0.015)
    return smax(d, -0.78 - p[:, 1], 0.01)  # flat bust base


PARTS = {"cranium": cranium, "snout": snout, "jaw": jaw, "horns": horns, "neck": neck,
         "ridge": ridge, "teeth": teeth, "eyelids": eyelids}
