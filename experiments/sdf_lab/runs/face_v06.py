# Adult male head, experiment 03: anatomical construction, no templates.
# Meters. Origin near head center, nasion at y=0. +z front, y up, x = subject's left.
# Proportions from anthropometric norms (head height ~232 mm, face n-me ~120 mm, nose n-sn ~53 mm, mouth ~52 mm).
from facelab import *

EPS, STEP, MAX_STEPS, AO = 1.5e-4, 0.6, 320, 0.006
VIEW_NAMES = ["front", "side (from +x)", "3/4", "back", "side (from -x)", "top"]

# --- front midline profile, top to bottom: (y, z) ---
MID = {
    "trichion":    (0.068, 0.088),
    "forehead":    (0.040, 0.097),
    "glabella":    (0.012, 0.100),
    "nasion":      (0.000, 0.094),
    "rhinion":     (-0.022, 0.104),
    "pronasale":   (-0.045, 0.120),
    "columella":   (-0.050, 0.112),
    "subnasale":   (-0.053, 0.101),
    "labrale_sup": (-0.063, 0.105),
    "stomion":     (-0.0735, 0.100),
    "labrale_inf": (-0.083, 0.102),
    "sublabiale":  (-0.096, 0.092),
    "pogonion":    (-0.107, 0.097),
    "gnathion":    (-0.116, 0.091),
    "menton":      (-0.121, 0.080),
    "cervical":    (-0.124, 0.042),
}
# --- back midline, top to bottom: the occiput overhangs the neck ---
BACK = {
    "occiput_upper":  (0.093, -0.074),
    "opisthocranion": (0.030, -0.103),
    "inion":          (-0.015, -0.088),
    "nape":           (-0.070, -0.060),
    "neck_back":      (-0.150, -0.058),
}
PROFILE = list(MID)

# --- bilateral surface landmarks (left side) ---
SIDE = {
    "frontotemporale_L": (0.052, 0.052, 0.078),
    "eurion_L":          (0.077, 0.040, -0.005),
    "supraorbitale_L":   (0.034, 0.015, 0.093),
    "orbitale_L":        (0.033, -0.018, 0.082),
    "zygion_L":          (0.069, -0.012, 0.040),
    "cheek_L":           (0.050, -0.045, 0.072),
    "alare_L":           (0.0175, -0.047, 0.097),
    "cheilion_L":        (0.026, -0.0745, 0.089),
    "jaw_mid_L":         (0.042, -0.105, 0.045),
    "gonion_L":          (0.054, -0.085, -0.002),
}
LANDMARKS = with_mirrors({**{k: (0, y, z) for k, (y, z) in {**MID, **BACK}.items()}, **SIDE})

# --- eye (left): center (x, y) only — depth is found from the skin; canthi and lid points (x, y) ---
EYE = dict(center=(0.031, -0.002), r=0.012,
           en=(0.0165, -0.0015), ex=(0.046, 0.001), up=(0.030, 0.0045), lo=(0.032, -0.006))
TRAGION = (-0.020, -0.005)  # (y, z): x is snapped onto the head


def base(p):
    """Blockout: cranium, frontal bone, lower face, chin, jaw, cheekbones, neck, nose blank."""
    q = mirror_x(p)
    d = ellipsoid(p, (0, 0.035, -0.005), (0.077, 0.082, 0.098))                  # cranium
    d = smin(d, ellipsoid(p, (0, 0.026, 0.050), (0.062, 0.050, 0.046)), 0.02)   # frontal bone
    d = smin(d, ellipsoid(p, (0, -0.055, 0.036), (0.054, 0.066, 0.060)), 0.03)  # lower face mass
    d = smin(d, ellipsoid(p, (0, -0.068, 0.064), (0.030, 0.020, 0.032)), 0.02)  # maxilla / teeth arch
    d = smin(d, ellipsoid(p, (0, -0.083, 0.090), (0.020, 0.008, 0.010)), 0.01)  # lower lip mass
    d = smin(d, ellipsoid(p, (0, -0.106, 0.073), (0.021, 0.017, 0.020)), 0.02)  # chin
    d = smin(d, capsule(q, (0.038, -0.086, -0.002), (0.016, -0.112, 0.074), 0.010), 0.025)  # jaw line
    d = smin(d, ellipsoid(q, (0.053, -0.012, 0.042), (0.016, 0.014, 0.022)), 0.02)  # cheekbones
    d = smin(d, ellipsoid(q, (0.040, -0.045, 0.054), (0.018, 0.020, 0.016)), 0.02)  # cheek pads
    d = smin(d, capsule(p, (0, -0.06, -0.008), (0, -0.20, -0.005), 0.054), 0.03)  # neck: under the skull, not behind it
    d = smin(d, capsule(p, (0, -0.02, 0.093), (0, -0.044, 0.108), 0.006, 0.010), 0.01)  # nose blank
    return smax(d, -0.19 - p[:, 1], 0.005)


prof_pts = list(spline([LANDMARKS[k] for k in PROFILE], 0.006))
back_pts = list(spline([LANDMARKS[k] for k in BACK], 0.012))
side_pts = [v for k, v in LANDMARKS.items() if k not in MID and k not in BACK]
fitted, _d0 = fit(base, prof_pts + back_pts + side_pts,
                  radius=[0.028] * len(prof_pts) + [0.045] * len(back_pts) + [0.04] * len(side_pts))
FIT_CORRECTIONS = dict(zip(LANDMARKS, base(np.array(list(LANDMARKS.values())))))

M = lambda y, z, x=0.0: (x, y, z)  # midline point helper
OPS = [
    # nose: nostrils are ~11 mm long, ~6.5 mm wide, angled; wings flare so they read from the front
    ("nostril", "carve", [(0.0085, -0.0515, 0.103)], (0.0032, 0.0028, 0.0055), 0.001, True),
    ("alar_wing", "inflate", [(0.0165, -0.048, 0.097)], 0.0025, 0.009, True),
    ("alar_groove", "crease", [(0.012, -0.040, 0.104), (0.021, -0.043, 0.098), (0.023, -0.051, 0.096), (0.016, -0.055, 0.099)],
     0.0010, 0.004, True),
    # cheeks / folds
    ("nasolabial", "crease", [(0.023, -0.046, 0.094), (0.030, -0.060, 0.090), (0.035, -0.078, 0.082)], 0.0012, 0.009, True),
    ("cheek_mass", "inflate", [(0.042, -0.058, 0.082)], 0.0015, 0.016, True),
    # mouth
    ("upper_lip", "ridge", [(0.02, -0.070, 0.094), M(-0.068, 0.103), (-0.02, -0.070, 0.094)], 0.0012, 0.004, False),
    ("lower_lip", "ridge", [(0.018, -0.079, 0.094), M(-0.081, 0.101), (-0.018, -0.079, 0.094)], 0.0018, 0.005, False),
    ("vermilion", "ridge", [(0.025, -0.072, 0.090), (0.010, -0.0625, 0.101), (0.004, -0.061, 0.104), M(-0.0625, 0.105),
                            (-0.004, -0.061, 0.104), (-0.010, -0.0625, 0.101), (-0.025, -0.072, 0.090)], 0.0005, 0.0035, False),
    ("mouth_slit", "groove", [(0.026, -0.0745, 0.089), (0.012, -0.0745, 0.097), M(-0.0735, 0.100), (-0.012, -0.0745, 0.097),
                              (-0.026, -0.0745, 0.089)], 0.0009, 0.0006, False),
    ("mouth_corner", "crease", [(0.022, -0.0745, 0.091), (0.029, -0.075, 0.087)], 0.0015, 0.004, True),
    ("philtrum", "crease", [M(-0.054, 0.101), M(-0.061, 0.104)], 0.0006, 0.009, False),
    ("philtral_column", "ridge", [(0.004, -0.054, 0.101), (0.0045, -0.061, 0.104)], 0.0004, 0.003, True),
    ("mentolabial", "crease", [(0.015, -0.094, 0.092), M(-0.096, 0.093), (-0.015, -0.094, 0.092)], 0.0010, 0.008, False),
    ("chin_pad", "inflate", [M(-0.107, 0.097)], 0.001, 0.012, False),
    # eyes / brow
    ("upper_lid_crease", "crease", [(0.019, 0.004, 0.086), (0.030, 0.010, 0.087), (0.044, 0.007, 0.082)], 0.0008, 0.004, True),
    ("brow_ridge", "ridge", [(0.015, 0.014, 0.095), (0.035, 0.017, 0.093), (0.050, 0.012, 0.084)], 0.0010, 0.011, True),
]

face = sculpt(fitted, OPS)
face = eye(face, **EYE, lid=0.0009)
sdf = ear(face, TRAGION)
