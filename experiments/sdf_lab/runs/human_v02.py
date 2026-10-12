# Adult male body, A-pose, built from nothing. Meters, floor at y=0 (shifted to -0.9 for the lab), +z = front.
# Skeleton first: every body part hangs off a named joint, so nothing can float.
from lab import *

V = lambda *a: np.array(a, float)
unit = lambda v: v / np.linalg.norm(v)

# --- skeleton (left side, x>0; right side comes from mirror_x) ---
SHOULDER = V(0.185, 1.43, -0.01)
ARM_DIR = unit(V(0.64, -0.77, 0.0))                 # A-pose, ~40 deg from the body
ELBOW = SHOULDER + 0.29 * ARM_DIR
FORE_DIR = unit(V(0.60, -0.77, 0.10))               # slight forward bend
WRIST = ELBOW + 0.255 * FORE_DIR
HIP = V(0.09, 0.93, 0.0)
KNEE = V(0.10, 0.50, 0.015)
ANKLE = V(0.095, 0.085, -0.02)


def head(p):
    q = mirror_x(p)
    d = ellipsoid(p, (0, 1.655, -0.012), (0.079, 0.108, 0.100))                # cranium
    d = smin(d, ellipsoid(p, (0, 1.580, 0.022), (0.062, 0.062, 0.072)), 0.04)  # jaw / lower face
    d = smin(d, ellipsoid(p, (0, 1.528, 0.064), (0.024, 0.018, 0.022)), 0.03)  # chin
    d = smin(d, ellipsoid(q, (0.048, 1.628, 0.052), (0.022, 0.014, 0.02)), 0.03)   # cheekbones (subtle)
    d = smin(d, ellipsoid(p, (0, 1.672, 0.070), (0.052, 0.012, 0.018)), 0.03)      # brow ridge (soft)
    d = smin(d, capsule(p, (0, 1.655, 0.086), (0, 1.612, 0.104), 0.007, 0.010), 0.012)  # nose bridge
    d = smin(d, sphere(p, (0, 1.610, 0.102), 0.012), 0.01)                           # nose tip
    d = smin(d, ellipsoid(q, (0.013, 1.606, 0.092), (0.009, 0.007, 0.008)), 0.006)   # nostril wings
    d = smin(d, ellipsoid(q, (0.077, 1.635, -0.012), (0.007, 0.027, 0.016)), 0.006)  # ears, close to the head
    d = smin(d, ellipsoid(p, (0, 1.583, 0.088), (0.024, 0.006, 0.010)), 0.006)       # upper lip
    d = smin(d, ellipsoid(p, (0, 1.571, 0.084), (0.021, 0.007, 0.010)), 0.006)       # lower lip
    d = cut(d, sphere(q, (0.032, 1.643, 0.090), 0.016), 0.01)                        # shallow eye sockets
    d = np.minimum(d, sphere(q, (0.032, 1.642, 0.070), 0.0125))                      # eyeballs, set back
    d = smin(d, ellipsoid(q, (0.032, 1.650, 0.076), (0.016, 0.007, 0.009)), 0.004)   # upper lids
    d = smin(d, ellipsoid(q, (0.032, 1.634, 0.075), (0.015, 0.004, 0.008)), 0.004)   # lower lids
    return d


def neck(p):
    q = mirror_x(p)
    d = capsule(p, (0, 1.45, -0.02), (0, 1.57, 0.0), 0.054, 0.048)
    d = smin(d, capsule(q, (0.035, 1.585, -0.005), (0.015, 1.465, 0.04), 0.014), 0.015)  # sternocleidomastoid
    return d


def torso(p):
    q = mirror_x(p)
    d = ellipsoid(p, (0, 1.29, -0.005), (0.145, 0.19, 0.105))                     # ribcage
    d = smin(d, ellipsoid(q, (0.065, 1.335, 0.06), (0.075, 0.055, 0.04)), 0.03)   # pecs
    d = smin(d, ellipsoid(p, (0, 1.10, 0.01), (0.13, 0.16, 0.095)), 0.08)         # abdomen
    d = smin(d, ellipsoid(q, (0.10, 1.24, -0.035), (0.06, 0.13, 0.065)), 0.05)    # lats
    d = smin(d, ellipsoid(p, (0, 0.93, 0.0), (0.15, 0.11, 0.105)), 0.08)          # pelvis
    d = smin(d, ellipsoid(q, (0.075, 0.89, -0.055), (0.085, 0.10, 0.075)), 0.04)  # glutes
    d = smin(d, capsule(q, (0.03, 1.50, -0.04), (0.16, 1.44, -0.03), 0.03, 0.022), 0.05)  # traps
    return d


def arm(p):
    q = mirror_x(p)
    d = capsule(q, SHOULDER - V(0.01, -0.01, 0), SHOULDER + 0.35 * (ELBOW - SHOULDER), 0.05, 0.042)  # deltoid cap
    d = smin(d, capsule(q, SHOULDER, ELBOW, 0.042, 0.032), 0.05)                        # upper arm
    d = smin(d, capsule(q, ELBOW + 0.05 * FORE_DIR, ELBOW + 0.45 * (WRIST - ELBOW), 0.037, 0.03), 0.04)  # forearm mass
    d = smin(d, capsule(q, ELBOW, WRIST, 0.033, 0.023), 0.04)                           # forearm
    return d


def hand(p):
    """Palm faces the thigh; fingers continue the forearm; spread across z, thumb in front."""
    q = mirror_x(p)
    f = FORE_DIR
    side = unit(V(0, 0, 1) - f * f[2])               # front-back axis across the palm
    knuckle = WRIST + 0.095 * f
    d = capsule(q, WRIST, knuckle, 0.024)
    d = smin(d, capsule(q, WRIST + side * 0.012, knuckle + side * 0.016, 0.02), 0.01)
    d = smin(d, capsule(q, WRIST - side * 0.012, knuckle - side * 0.016, 0.02), 0.01)
    for s, L in ((0.026, 0.085), (0.009, 0.095), (-0.009, 0.09), (-0.026, 0.072)):  # index..pinky
        base = knuckle + side * s
        d = smin(d, capsule(q, base, base + f * L + side * s * 0.15, 0.0105, 0.008), 0.006)
    thumb_base = WRIST + 0.02 * f + side * 0.025
    d = smin(d, capsule(q, thumb_base, thumb_base + unit(f + side * 1.2) * 0.065, 0.012, 0.009), 0.01)
    return d


def leg(p):
    q = mirror_x(p)
    d = capsule(q, HIP, KNEE, 0.085, 0.05)                                               # thigh
    d = smin(d, ellipsoid(q, HIP + 0.45 * (KNEE - HIP) + V(0.0, 0, 0.035), (0.06, 0.13, 0.05)), 0.04)  # quads
    d = smin(d, sphere(q, KNEE + V(0, 0.01, 0.035), 0.03), 0.02)                         # kneecap
    d = smin(d, capsule(q, KNEE, ANKLE, 0.048, 0.03), 0.03)                              # shin
    d = smin(d, ellipsoid(q, (0.10, 0.36, -0.03), (0.045, 0.09, 0.045)), 0.03)           # calf
    return d


def foot(p):
    q = mirror_x(p)
    d = capsule(q, ANKLE, (0.10, 0.04, -0.04), 0.032, 0.035)                             # heel
    d = smin(d, capsule(q, (0.10, 0.035, -0.045), (0.105, 0.03, 0.12), 0.035, 0.03), 0.03)  # foot body
    d = smin(d, capsule(q, (0.10, 0.022, 0.12), (0.11, 0.018, 0.17), 0.022, 0.016), 0.015)  # toes
    return smax(d, -p[:, 1], 0.005)                                                      # flat sole on the floor


def body(p):
    d = smin(head(p), neck(p), 0.03)
    d = smin(d, torso(p), 0.05)
    d = smin(d, arm(p), 0.04)
    d = smin(d, hand(p), 0.015)
    d = smin(d, leg(p), 0.05)
    d = smin(d, foot(p), 0.03)
    return d


SHIFT = V(0, 0.9, 0)
sdf = lambda p: body(p + SHIFT)
PARTS = {k: (lambda f: lambda p: f(p + SHIFT))(f) for k, f in
         {"head": head, "neck": neck, "torso": torso, "arm": arm, "hand": hand, "leg": leg, "foot": foot}.items()}
