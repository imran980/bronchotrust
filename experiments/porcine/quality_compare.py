"""Fair raw-frame-quality comparison between two airway videos over a chosen window.
Measures, on identically-sampled frames: sharpness (Laplacian var), brightness (mean L*),
contrast (std L*), specular-blowout %, and SIFT feature density on the CLAHE'd frame (what
actually feeds COLMAP). No reconstruction. Args: tag:video:lo:hi:stride ... """
import sys, cv2, numpy as np
sift = cv2.SIFT_create(nfeatures=8192, contrastThreshold=0.005)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
print(f"{'video':8} {'sharp(Lap)':>11} {'bright':>7} {'contrast':>9} {'specular%':>9} {'SIFT/img':>9} {'valid%':>7}")
for tok in sys.argv[1:]:
    tag, v, lo, hi, st = tok.split(":"); lo, hi, st = int(lo), int(hi), int(st)
    cap = cv2.VideoCapture(v); fi = 0
    sh, br, ct, sp, kp, nval, ntot = [], [], [], [], [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok or fi > hi: break
        if lo <= fi <= hi and (fi - lo) % st == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); L = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)[:, :, 0]
            ntot += 1
            if L.mean() * 100 / 255 < 12: fi += 1; continue          # skip dark (pipeline drops these)
            nval += 1
            sh.append(cv2.Laplacian(g, cv2.CV_64F).var())
            br.append(L.mean() * 100 / 255); ct.append(L.std())
            sp.append(100.0 * (g > 245).mean())
            gc = clahe.apply(g); kp.append(len(sift.detect(gc, None)))
        fi += 1
    cap.release()
    f = lambda a: np.median(a) if a else 0
    print(f"{tag:8} {f(sh):11.0f} {f(br):7.1f} {f(ct):9.1f} {f(sp):9.2f} {int(f(kp)):9d} {100*nval/max(ntot,1):6.0f}%")
