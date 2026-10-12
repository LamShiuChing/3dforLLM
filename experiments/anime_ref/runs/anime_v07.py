# Anime girl head from a reference sheet (reference.png, Anima seed 1). Same face vocabulary as exp03.
# Landmarks were read off grid overlays: 1 ref px = 0.564 mm; y=0 at the eye line, x=0 face center, z=0 head center (side view).
# x = subject's left, +z front. Stylized proportions: realistic anatomy lint is switched off.
import facelab
from facelab import *

facelab.TYPICAL.clear()  # anime proportions deliberately break adult anatomy sizes
EPS, STEP, MAX_STEPS, AO = 1.5e-4, 0.6, 320, 0.006
VIEW_NAMES = ["front", "side (from +x)", "3/4", "back", "side (from -x)", "top"]

px = lambda u: u * 0.000564  # reference pixels -> meters

# --- front midline profile from the side view, top to bottom: (y, z) ---
MID = {
    "hairline":     (px(350 - 170), px(1080 - 945)),
    "forehead_up":  (px(350 - 200), px(1080 - 925)),
    "forehead":     (px(350 - 250), px(1080 - 908)),
    "glabella":     (px(350 - 300), px(1080 - 912)),
    "nasion":       (px(350 - 340), px(1080 - 918)),
    "nose_mid":     (px(350 - 370), px(1080 - 905)),
    "pronasale":    (px(350 - 398), px(1080 - 890)),
    "subnasale":    (px(350 - 420), px(1080 - 906)),
    "upper_lip":    (px(350 - 440), px(1080 - 915)),
    "stomion":      (px(350 - 455), px(1080 - 922)),
    "lower_lip":    (px(350 - 465), px(1080 - 922)),
    "chin_front":   (px(350 - 490), px(1080 - 928)),
    "menton":       (px(350 - 507), px(1080 - 942)),
    "under_jaw":    (px(350 - 500), px(1080 - 1005)),
    "throat":       (px(350 - 560), px(1080 - 1025)),
}
BACK = {  # skull under the hair (hair is ~8 px thick on the sheet)
    "skull_top":    (px(350 - 128), 0.0),
    "back_upper":   (px(350 - 190), px(1080 - 1225)),
    "back":         (px(350 - 300), px(1080 - 1245)),
    "occiput_low":  (px(350 - 400), px(1080 - 1215)),
    "nape":         (px(350 - 460), px(1080 - 1150)),
    "neck_back":    (px(350 - 560), px(1080 - 1140)),
}
# corrections from the reference-compare sense (z, meters): + = move forward
NUDGE = {"forehead": -0.001, "glabella": 0.0015, "nasion": 0.0015, "nose_mid": -0.003, "pronasale": -0.0035,
         "upper_lip": 0.002, "stomion": 0.002, "lower_lip": 0.002, "chin_front": -0.003, "menton": -0.003,
         "under_jaw": -0.002, "throat": -0.0035}
MID = {k: (y, z + NUDGE.get(k, 0)) for k, (y, z) in MID.items()}
PROFILE = list(MID)

# --- front-view silhouette as a 2D curve (half-width per height), read off the grid; the engine trims to it ---
FRONT_OUTLINE = [(px(350 - ypx), px(xpx - 446)) for ypx, xpx in
                 [(200, 606), (240, 580), (260, 567), (280, 573), (300, 575), (330, 570), (350, 564), (400, 562), (425, 559), (440, 554), (460, 542), (480, 520),
                  (486, 518), (600, 518)]]
SIDE = {"cheilion_L": (px(473 - 446), px(350 - 457), 0.084)}  # mouth corner sits on the surface
LANDMARKS = with_mirrors({**{k: (0, y, z) for k, (y, z) in {**MID, **BACK}.items()}, **SIDE})

# --- big anime eye (left = image right on the front view) ---
EYE = dict(center=(px(510 - 446), 0.0), r=0.019,
           en=(px(488 - 446), px(350 - 352)), ex=(px(555 - 446), px(350 - 340)),
           up=(px(515 - 446), px(350 - 328)), lo=(px(505 - 446), px(350 - 372)))
TRAGION = (px(350 - 375), px(1080 - 1100))


def base(p):
    """Blockout: wide round cranium, small V-shaped lower face, tiny nose, slim neck."""
    q = mirror_x(p)
    d = ellipsoid(p, (0, 0.030, 0.0015), (0.072, 0.098, 0.096))                  # cranium (temple skin measured 73 mm)
    d = smin(d, ellipsoid(p, (0, 0.062, 0.040), (0.058, 0.050, 0.052)), 0.03)   # frontal bone: forehead forward
    d = smin(d, ellipsoid(p, (0, -0.042, 0.030), (0.050, 0.048, 0.058)), 0.03)  # lower face (narrower)
    d = smin(d, ellipsoid(p, (0, -0.058, 0.072), (0.022, 0.014, 0.016)), 0.015)  # mouth area: lips sit forward
    d = smin(d, capsule(q, (0.050, -0.046, 0.005), (0.011, -0.085, 0.072), 0.008), 0.02)  # V jaw line (narrower)
    d = smin(d, ellipsoid(p, (0, -0.080, 0.068), (0.020, 0.010, 0.014)), 0.015)  # small chin
    d = smin(d, capsule(p, (0, -0.000, 0.088), (0, -0.027, 0.100), 0.004, 0.007), 0.01)  # nose blank
    d = smin(d, capsule(p * (1, 1, 1.15), (0, -0.04, -0.002), (0, -0.20, -0.002), 0.040), 0.03)  # neck: 80 mm wide, ~70 mm deep
    return smax(d, -0.19 - p[:, 1], 0.005)


prof_pts = list(spline([LANDMARKS[k] for k in PROFILE], 0.006))
back_pts = list(spline([LANDMARKS[k] for k in BACK], 0.010))
side_pts = [v for k, v in LANDMARKS.items() if k not in MID and k not in BACK]
fitted, _d0 = fit(base, prof_pts + back_pts + side_pts,
                  radius=[0.025] * len(prof_pts) + [0.045] * len(back_pts) + [0.035] * len(side_pts))
FIT_CORRECTIONS = dict(zip(LANDMARKS, base(np.array(list(LANDMARKS.values())))))

M = lambda y, z, x=0.0: (x, y, z)
OPS = [
    ("nostril", "carve", [(0.0035, px(350 - 415), px(1080 - 902))], (0.0016, 0.0010, 0.0022), 0.001, True),
    ("mouth_slit", "groove", [(px(473 - 446), px(350 - 458), 0.084), (0.007, px(350 - 456), 0.088), M(px(350 - 455), px(1080 - 922)),
                              (-0.007, px(350 - 456), 0.088), (-px(473 - 446), px(350 - 458), 0.084)], 0.0007, 0.0006, False),
    ("lower_lip", "ridge", [(0.010, px(350 - 463), 0.087), M(px(350 - 465), px(1080 - 922)), (-0.010, px(350 - 463), 0.087)],
     0.0008, 0.004, False),
    ("brow_ridge", "ridge", [(0.022, px(350 - 298), 0.090), (0.040, px(350 - 292), 0.085), (0.058, px(350 - 292), 0.070)],
     0.0006, 0.008, True),
]

face = sculpt(front_outline(fitted, FRONT_OUTLINE, k=0.012), OPS)
face = eye(face, **EYE, lid=0.0008, recess=0.001)
sdf = ear(face, TRAGION, flare=0.008, size=0.78)
