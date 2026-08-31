"""Is a reconstruction a TUBE (const radius along axis) or a CONE (radius grows with axial depth)?
Also tests the mechanism: did the camera physically TRAVERSE the structure (cameras spread along the
whole axial extent = tube), or sit at one end imaging the wall AHEAD (cameras clustered at narrow end,
points fan out ahead = cone from forward monocular motion). Args: dense.ply sparse_dir TAG"""
import sys, numpy as np, pycolmap, open3d as o3d
PLY, SPARSE, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
r = pycolmap.Reconstruction(SPARSE)
C = np.array([np.asarray(im.projection_center()) for im in r.images.values()])
# optical axis of each cam (viewing direction) to see if motion is ALONG the look direction
look = np.array([np.asarray(im.cam_from_world().rotation.matrix()[2]) for im in r.images.values()])
P = np.asarray(o3d.io.read_point_cloud(PLY).points)
m = np.median(P, 0); dd = np.linalg.norm(P - m, axis=1)
P = P[dd < np.median(dd) + 4 * np.median(np.abs(dd - np.median(dd)))]

Cc = C - C.mean(0); _, _, Vt = np.linalg.svd(Cc, full_matrices=False); ax = Vt[0]
if (Cc @ ax).mean() < (Cc @ ax)[0]:  # orient axis along direction of camera travel (start->end)
    pass
order = np.argsort(Cc @ ax)
if (Cc @ ax)[order][-1] < (Cc @ ax)[order][0]: ax = -ax
Pc = P - C.mean(0)
pa = Pc @ ax; prad = np.linalg.norm(Pc - np.outer(pa, ax), axis=1)     # axial coord & perpendicular radius
ca = Cc @ ax

# 1) RADIUS PROFILE along axis: bin axial coord, median radius per bin, linear fit slope = cone half-slope
lo, hi = np.percentile(pa, 3), np.percentile(pa, 97)
edges = np.linspace(lo, hi, 11); mids = 0.5 * (edges[:-1] + edges[1:]); prof = []
for a, b in zip(edges[:-1], edges[1:]):
    s = (pa >= a) & (pa < b); prof.append(np.median(prad[s]) if s.sum() > 30 else np.nan)
prof = np.array(prof); ok = ~np.isnan(prof)
slope, intercept = np.polyfit(mids[ok], prof[ok], 1)
r_lo = np.median(prad[(pa >= lo) & (pa < lo + (hi - lo) / 5)])
r_hi = np.median(prad[(pa <= hi) & (pa > hi - (hi - lo) / 5)])

# 2) TRAVERSAL: do cameras span the axial extent of the points, or cluster at one end?
cam_span = ca.max() - ca.min(); pt_span = hi - lo
# fraction of points that lie AHEAD of the last camera (forward-imaged wall)
ahead = (pa > ca.max()).mean()
behind = (pa < ca.min()).mean()
# where do cameras sit within the point axial range? 0=narrow/start end, 1=wide/far end
cam_center_frac = ((ca.mean() - lo) / (pt_span + 1e-9))

# 3) is camera motion ALONG the optical axis? (forward degeneracy) mean |cos| between step dirs and look
steps = np.diff(C[np.argsort(ca)], axis=0); steps /= np.linalg.norm(steps, axis=1, keepdims=True) + 1e-9
lookn = look / (np.linalg.norm(look, axis=1, keepdims=True) + 1e-9)
along = np.abs((steps * lookn[np.argsort(ca)][:-1]).sum(1)).mean()

print(f"=== {TAG} ===")
print(f"radius profile (median r at 10 axial bins, narrow->far):")
print("  " + "  ".join(f"{p:.2f}" if not np.isnan(p) else "  --" for p in prof))
print(f"  r_near={r_lo:.2f}  r_far={r_hi:.2f}  ratio={r_hi/max(r_lo,1e-6):.2f}x   taper slope dr/da={slope:.3f}")
print(f"    -> {'CONE (radius grows >1.5x along axis)' if r_hi/max(r_lo,1e-6)>1.5 else 'TUBE-like (radius ~const)'}")
print(f"traversal: cam axial travel={cam_span:.2f}  point axial extent={pt_span:.2f}  cam/point={cam_span/max(pt_span,1e-6):.2f}")
print(f"  points AHEAD of last camera: {ahead:.0%}   BEHIND first camera: {behind:.0%}")
print(f"  cameras sit at frac {cam_center_frac:.2f} of the point range (0=narrow end, 1=far end)")
print(f"  camera-step vs optical-axis alignment |cos|={along:.2f}  (1.0 = pure forward/axial motion = monocular degeneracy)")
