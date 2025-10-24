import open3d as o3d
import numpy as np
import json
import os

try:
    import pymeshfix

    HAS_PYMESHFIX = True
except ImportError:
    HAS_PYMESHFIX = False
    print("[WARN] PyMeshFix not installed. Watertight repair will be skipped.")


def auto_repair_watertight(mesh: o3d.geometry.TriangleMesh):
    """
    Automatically fills holes and fixes non-watertight areas using PyMeshFix.
    """
    if not HAS_PYMESHFIX:
        print("[INFO] Skipping watertight repair (PyMeshFix missing).")
        return mesh

    print("[INFO] Performing watertight repair using PyMeshFix...")
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)

    mf = pymeshfix.MeshFix(v, f)
    mf.repair(verbose=False)

    v2, f2 = mf.v, mf.f
    fixed = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v2), o3d.utility.Vector3iVector(f2)
    )
    fixed.compute_vertex_normals()
    return fixed


def analyze_and_visualize(mesh_path: str, report_path: str = None):
    print(f"[INFO] Loading: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    # ============================================================
    # Remove degenerate triangles
    # ============================================================
    def area(v0, v1, v2):
        return np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2

    areas = np.array([area(verts[a], verts[b], verts[c]) for a, b, c in tris])
    mask = areas > 1e-7
    clean_tris = tris[mask]
    mesh.triangles = o3d.utility.Vector3iVector(clean_tris)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    # ============================================================
    # Attempt watertight repair if needed
    # ============================================================
    before_wt = True
    if not before_wt:
        print("[INFO] Mesh is NOT watertight — attempting repair...")
        mesh = auto_repair_watertight(mesh)
    after_wt = mesh.is_watertight()

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    colors = np.full_like(verts, [0.7, 0.7, 0.7])  # base gray

    report = {}

    # ============================================================
    # 1️⃣ Duplicated vertices  (red)
    # ============================================================
    unique, counts = np.unique(np.round(verts, 5), axis=0, return_counts=True)
    duplicates = unique[counts > 1]
    dup_indices = []
    for dup in duplicates:
        idx = np.where(np.all(np.isclose(verts, dup, atol=1e-5), axis=1))[0]
        colors[idx] = [1.0, 0.0, 0.0]
        dup_indices.extend(idx)
    report["duplicated_vertices"] = len(dup_indices)

    # ============================================================
    # 2️⃣ Degenerate triangles  (green)
    # ============================================================
    def tri_area(v0, v1, v2):
        return np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2

    areas = np.array([tri_area(verts[a], verts[b], verts[c]) for a, b, c in tris])
    degenerate = np.where(areas < 1e-7)[0]
    deg_verts = np.unique(tris[degenerate].flatten())
    colors[deg_verts] = [0.0, 1.0, 0.0]
    report["degenerate_faces"] = len(degenerate)

    # ============================================================
    # 3️⃣ Non-manifold edges  (blue)
    # ============================================================
    nm_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    if len(nm_edges) > 0:
        nm_vertices = np.unique(nm_edges.flatten())
        colors[nm_vertices] = [0.2, 0.6, 1.0]
    report["non_manifold_edges"] = int(len(nm_edges))

    # ============================================================
    # 4️⃣ Small isolated components  (purple)
    # ============================================================
    tri_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    small_clusters = np.where(cluster_n_triangles < 100)[0]
    small_tris_idx = np.where(np.isin(tri_clusters, small_clusters))[0]
    small_verts = np.unique(tris[small_tris_idx].flatten())
    colors[small_verts] = [0.8, 0.3, 1.0]
    report["small_clusters"] = len(small_clusters)

    # ============================================================
    # 5️⃣ Outlier vertices  (yellow)
    # ============================================================
    centroid = verts.mean(axis=0)
    distances = np.linalg.norm(verts - centroid, axis=1)
    threshold = np.mean(distances) * 3.0
    outliers = np.where(distances > threshold)[0]
    colors[outliers] = [1.0, 1.0, 0.0]
    report["outliers"] = len(outliers)

    # ============================================================
    # 6️⃣ Watertight boundary edges  (cyan)
    # ============================================================
    edge_count = {}
    for tri in tris:
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            e_sorted = tuple(sorted(e))
            edge_count[e_sorted] = edge_count.get(e_sorted, 0) + 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    boundary_vertices = np.unique(np.array(boundary_edges).flatten()).astype(int)
    colors[boundary_vertices] = [0.0, 1.0, 1.0]
    report["boundary_edges"] = len(boundary_edges)

    # ============================================================
    # Global mesh info
    # ============================================================
    report["was_watertight_before"] = bool(before_wt)
    report["is_watertight_after"] = bool(after_wt)
    report["is_edge_manifold"] = bool(mesh.is_edge_manifold())
    report["is_vertex_manifold"] = bool(mesh.is_vertex_manifold())
    report["vertices"] = int(len(mesh.vertices))
    report["faces"] = int(len(mesh.triangles))

    # ============================================================
    # Visualization
    # ============================================================
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    print(json.dumps(report, indent=2))
    o3d.visualization.draw_geometries([mesh], window_name="QA + Watertight Repair")

    # ============================================================
    # Save JSON report
    # ============================================================
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Report saved to {report_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(
        BASE_DIR, "data/input/bunny/reconstruction/bun_zipper.ply"
    )
    output_path = os.path.join(BASE_DIR, "data/output/qa_report.json")
    analyze_and_visualize(mesh_path=input_path, report_path=output_path)
