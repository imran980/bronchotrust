"""Canonical figure style for the CMIG pediatric-airway paper.
IMPORT THIS in every figure script so colours, fonts, and conventions stay identical:

    from figstyle import apply_style, PATIENT, ROLE, DEPTH_CMAP
    apply_style()
    ...
    ax.plot(..., color=PATIENT["2-V2"])

Human-readable rationale + do/don't list lives in paper/FIGURE_STYLE.md. Palette is Okabe-Ito
(colour-vision-deficiency safe). Do NOT introduce colours outside this file.
"""

# ---- Okabe-Ito base palette (CVD-safe) ----
OKABE = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermillion":"#D55E00",
    "purple":    "#CC79A7",
    "sky":       "#56B4E9",
    "yellow":    "#F0E442",
    "black":     "#000000",
    "gray":      "#999999",
}

# ---- Per-patient identity colours (use in ANY multi-patient figure; NEVER reassign) ----
PATIENT = {
    "2-V2":  OKABE["blue"],
    "20-V1": OKABE["orange"],
    "30-V2": OKABE["green"],
    "9-V2":  OKABE["purple"],
    "50-V2": OKABE["sky"],
}

# ---- Semantic role colours (use in single-patient / role-based figures) ----
ROLE = {
    "ct":            "#333333",        # CT / ground truth curves (dark gray)
    "reconstruction":OKABE["vermillion"],  # the reconstruction, when contrasted against CT in ONE patient
    "inspiration":   OKABE["blue"],
    "expiration":    OKABE["orange"],
    "c3vd":          OKABE["sky"],     # external-benchmark bars
    "reference":     "#808080",        # identity / reference lines (dashed)
    "phantom":       "#666666",        # synthetic-phantom notes (italic gray)
    "accent_valid":  OKABE["blue"],    # validation-arm accents in the pipeline diagram
    "accent_dyn":    OKABE["orange"],
    "accent_ext":    OKABE["green"],
}

# 3D point-cloud / mesh depth colouring (proximal -> distal along the airway axis)
DEPTH_CMAP = "viridis"

# Neutral fills / ink
INK        = "#262626"   # primary text
INK_MUTED  = "#595959"   # secondary text / captions
SPINE      = "#666666"   # axis spines
GRID_ALPHA = 0.25
BOX_FILL   = "#ececec"   # neutral flow-box fill (pipeline diagram)

# Marker convention: validated data = FILLED markers; partial / low-confidence = OPEN markers.
PARTIAL_MARKER = dict(markerfacecolor="none", markeredgewidth=1.4)

FONT_STACK = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def apply_style():
    """Set global matplotlib rcParams. Call once at the top of each figure script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 190,
    })


def patient_color(pid):
    return PATIENT[pid]


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, alpha=GRID_ALPHA, lw=0.6)
