"""EVIDENCE 4 -- objective vs NFL-BA, side by side (equations, not prose). Typesets both objectives
and shows ours is the NFL-BA objective under three restrictions (fixed poses, SoR geometry,
point-light). NFL-BA characterized by the canonical near-field photometric bundle-adjustment
structure; exact paper notation may differ but the variable/objective structure is as shown."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OURS = (r"$E_{\mathrm{ours}}(\,r(\cdot),\ \rho,\ I\,)=\sum_i\sum_{p\in\Omega_i}"
        r"\left\| B_i(p)-\rho(x)\,\frac{\max(0,\ n(x)\!\cdot\!l_{ip})}{d_{ip}^{2}}\,I \right\|^2$")
NFLBA = (r"$E_{\mathrm{NFLBA}}(\,\{T_i\},\{X_j\},\Theta,\rho\,)=\sum_i\sum_{j}w_{ij}"
         r"\left\| B_i(\pi(T_i,X_j))-\rho_j\,\Phi\!(n_j\!\cdot\!l_{ij},\,d_{ij};\,\Theta)\right\|^2$")


def main():
    fig = plt.figure(figsize=(13, 8.5)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    T = ax.text
    T(0.5, 0.965, "EVIDENCE 4 — proposed objective vs NFL-BA (side by side)", ha="center", fontsize=15, weight="bold")
    T(0.03, 0.90, "PROPOSED (near-light photometric SoR fit):", fontsize=12, weight="bold", color="tab:red")
    T(0.05, 0.845, OURS, fontsize=12.5)
    T(0.05, 0.79, r"variables:  $r(\cdot)$ (SoR radius profile),  $\rho$ (albedo),  $I$ (light)."
                  "\n" r"FIXED (not optimized):  poses $\{T_i\}$ (COLMAP/Barbour),  intrinsics $K$.",
      fontsize=10.5, va="top")
    T(0.05, 0.735, r"$x=x(r;\,T_i,K,p)$: SoR surface point that pixel $p$ back-projects to; "
                  r"$d_{ip}=\| C_i-x\|,\ l_{ip}=(C_i-x)/d_{ip}$.", fontsize=9.5, va="top")
    T(0.03, 0.655, "NFL-BA (near-field-lighting bundle adjustment; canonical form):", fontsize=12, weight="bold", color="tab:blue")
    T(0.05, 0.60, NFLBA, fontsize=12.5)
    T(0.05, 0.545, r"variables:  poses $\{T_i\}$,  geometry $\{X_j\}$ (dense/surface),  light $\Theta$,  albedo $\rho_j$."
                  "\n" r"$\Phi$ = near-field light model: inverse-square $\times$ angular/spot falloff.  $\pi$ = projection.",
      fontsize=10.5, va="top")
    T(0.03, 0.455, "REDUCTION (ours = NFL-BA under 3 restrictions):", fontsize=12, weight="bold")
    T(0.05, 0.405,
      r"(1)  $\{T_i\}$ moved from VARIABLES $\to$ CONSTANTS   (COLMAP poses; not refined in the loop)"
      "\n\n" r"(2)  geometry $\{X_j\}\ \rightarrow\ $ restricted to a surface of revolution $S(r(\cdot))$   (low-dim tubular prior)"
      "\n\n" r"(3)  light $\Phi(n\!\cdot\!l,d;\Theta)\ \rightarrow\ (n\!\cdot\!l)\,I/d^{2}$   (drop angular/spot falloff $\Theta$; point source)",
      fontsize=11, va="top")
    T(0.05, 0.17,
      r"$\Rightarrow\quad E_{\mathrm{ours}}(r,\rho,I)\ =\ E_{\mathrm{NFLBA}}(\{T_i\},\{X_j\},\Theta,\rho)"
      r"\,|_{\,T_i\,\mathrm{fixed},\ X_j\in S(r),\ \Phi=I/d^2}$", fontsize=13)
    T(0.05, 0.075,
      "Same data term (near-light Lambertian photometric residual), same unknown TYPES (geometry, albedo, light).\n"
      "The ONLY structural change is dropping poses from the variable set (and specializing geometry/light).\n"
      "=> ours is a RESTRICTION of the NFL-BA objective, not a different inverse problem. "
      "(Test 3 shows the price of restriction (1): fixed noisy poses inject a +21% CSA bias that a\n"
      "pose-refining BA would partly absorb.)   [NFL-BA notation paraphrased; structure faithful.]",
      fontsize=9.5, va="top", color="0.15")
    fig.savefig("evidence4_nflba_equations.png", dpi=130, facecolor="white")
    print("wrote evidence4_nflba_equations.png")


if __name__ == "__main__":
    main()
