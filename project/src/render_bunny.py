"""
Render Stanford Bunny in 5 representations with consistent color.

Usage:
    python render_bunny.py

Output folder: bunny_renders/
"""

import open3d as o3d
import numpy as np
import urllib.request
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "bunny_renders"
WIDTH = 1600
HEIGHT = 1600
BACKGROUND = [1.0, 1.0, 1.0]

# Consistent color for all representations — slate blue, prints well
COLOR = np.array([0.45, 0.55, 0.75])
EDGE_COLOR = np.array([0.15, 0.20, 0.35])

BUNNY_URL = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj"
BUNNY_FILE = "bunny.obj"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_bunny():
    if not os.path.exists(BUNNY_FILE):
        print("Downloading Stanford Bunny...")
        urllib.request.urlretrieve(BUNNY_URL, BUNNY_FILE)
        print("Done.")
    else:
        print("Bunny already downloaded.")


# ---------------------------------------------------------------------------
# Load, normalize, orient
# ---------------------------------------------------------------------------
def load_mesh():
    mesh = o3d.io.read_triangle_mesh(BUNNY_FILE)
    mesh.compute_vertex_normals()

    center = mesh.get_center()
    mesh.translate(-center)
    scale = np.max(np.linalg.norm(np.asarray(mesh.vertices), axis=1))
    mesh.scale(1.0 / scale, center=[0, 0, 0])

    R = mesh.get_rotation_matrix_from_xyz((0.0, np.pi, 0.0))
    mesh.rotate(R, center=[0, 0, 0])

    return mesh


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
ZOOM = 0.55
FRONT = [0.0, 0.1, -1.0]
LOOKAT = [0.0, 0.0, 0.0]
UP = [0.0, 1.0, 0.0]


def apply_camera(vis, zoom=None):
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom if zoom is not None else ZOOM)
    ctr.set_front(FRONT)
    ctr.set_lookat(LOOKAT)
    ctr.set_up(UP)


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------
def render(geometry_list, filename, point_size=2.0, zoom=0.6):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=WIDTH, height=HEIGHT)

    opt = vis.get_render_option()
    opt.background_color = np.array(BACKGROUND)
    opt.point_size = point_size
    opt.light_on = True

    for geom in geometry_list:
        vis.add_geometry(geom)

    vis.poll_events()
    vis.update_renderer()
    apply_camera(vis, zoom=zoom)
    vis.poll_events()
    vis.update_renderer()

    out_path = os.path.join(OUTPUT_DIR, filename)
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 1. Point Cloud
# ---------------------------------------------------------------------------
def render_pointcloud(mesh):
    pcd = mesh.sample_points_uniformly(number_of_points=100_000)
    pcd.paint_uniform_color(COLOR)
    render([pcd], "bunny_pointcloud.png", point_size=1.5)


# ---------------------------------------------------------------------------
# 2. Mesh with wireframe overlay
# ---------------------------------------------------------------------------
def render_mesh(mesh):
    m = o3d.geometry.TriangleMesh(mesh)
    m.paint_uniform_color(COLOR)

    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(m)
    wireframe.paint_uniform_color(EDGE_COLOR)

    render([m, wireframe], "bunny_mesh.png")


# ---------------------------------------------------------------------------
# 3. Voxel Grid
#    VoxelGrid ignores uniform color — must set per-voxel color explicitly,
#    then render only the edge LineSet over a colored voxel mesh.
#    Workaround: build colored TriangleMesh cubes + edge LineSet.
# ---------------------------------------------------------------------------
def render_voxel(mesh):
    voxel_size = 0.055
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh, voxel_size=voxel_size
    )
    voxels = voxel_grid.get_voxels()

    all_vertices = []
    all_triangles = []
    all_edges_pts = []
    all_edges_lines = []
    tri_base = 0
    edge_base = 0

    # Unit cube faces (2 triangles per face, 6 faces)
    unit_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom  y=0
            [4, 6, 7],
            [4, 7, 5],  # top     y=1
            [0, 4, 5],
            [0, 5, 1],  # front   z=0
            [2, 3, 7],
            [2, 7, 6],  # back    z=1
            [0, 2, 6],
            [0, 6, 4],  # left    x=0
            [1, 5, 7],
            [1, 7, 3],  # right   x=1
        ]
    )

    unit_edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    for voxel in voxels:
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        origin = center - voxel_size / 2.0

        corners = np.array(
            [
                origin + voxel_size * np.array([dx, dy, dz])
                for dx in [0, 1]
                for dy in [0, 1]
                for dz in [0, 1]
            ]
        )

        all_vertices.extend(corners)
        all_triangles.extend(unit_faces + tri_base)
        tri_base += 8

        all_edges_pts.extend(corners)
        for a, b in unit_edges:
            all_edges_lines.append([edge_base + a, edge_base + b])
        edge_base += 8

    # Colored mesh
    voxel_mesh = o3d.geometry.TriangleMesh()
    voxel_mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
    voxel_mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))
    voxel_mesh.compute_vertex_normals()
    voxel_mesh.paint_uniform_color(COLOR)

    # Edge LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(all_edges_pts))
    line_set.lines = o3d.utility.Vector2iVector(np.array(all_edges_lines))
    line_set.paint_uniform_color(EDGE_COLOR)

    render([voxel_mesh, line_set], "bunny_voxel.png")


# ---------------------------------------------------------------------------
# 4. Octree
#    Open3D Octree renders leaf nodes as colored cubes automatically.
#    We paint the source point cloud with COLOR so leaf nodes inherit it.
# ---------------------------------------------------------------------------
def render_octree(mesh):
    pcd = mesh.sample_points_uniformly(number_of_points=60_000)
    pcd.paint_uniform_color(COLOR)

    octree = o3d.geometry.Octree(max_depth=5)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    render([octree], "bunny_octree.png")


# ---------------------------------------------------------------------------
# 5. SDF — distance field shell, blue=near surface, red=further
# ---------------------------------------------------------------------------
def render_sdf(mesh, output_path="bunny_sdf.png", N=150, threshold=0.035):
    mesh.compute_vertex_normals()

    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound

    padding = 0.08 * np.max(max_bound - min_bound)
    min_bound = min_bound - padding
    max_bound = max_bound + padding

    lin_x = np.linspace(min_bound[0], max_bound[0], N)
    lin_y = np.linspace(min_bound[1], max_bound[1], N)
    lin_z = np.linspace(min_bound[2], max_bound[2], N)

    xx, yy, zz = np.meshgrid(lin_x, lin_y, lin_z, indexing="ij")
    query_pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    query_tensor = o3d.core.Tensor(query_pts.astype(np.float32))
    signed_dist = scene.compute_signed_distance(query_tensor).numpy()

    mask = np.abs(signed_dist) < threshold
    pts = query_pts[mask]
    dists = signed_dist[mask]

    if len(pts) == 0:
        print(
            "SDF: no points found near zero level-set. Increase threshold or check mesh scale."
        )
        return

    max_abs = np.max(np.abs(dists)) + 1e-8
    strength = np.abs(dists) / max_abs

    colors = np.ones((len(dists), 3), dtype=np.float64)

    inside = dists < 0
    outside = dists > 0

    colors[inside] = np.column_stack(
        [
            0.2 + 0.5 * strength[inside],
            0.35 + 0.35 * (1.0 - strength[inside]),
            1.0 * np.ones(np.sum(inside)),
        ]
    )

    colors[outside] = np.column_stack(
        [
            1.0 * np.ones(np.sum(outside)),
            0.25 + 0.45 * (1.0 - strength[outside]),
            0.25 + 0.45 * (1.0 - strength[outside]),
        ]
    )

    near_surface = np.abs(dists) < threshold * 0.15
    colors[near_surface] = np.array([1.0, 1.0, 1.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    render([pcd], output_path, point_size=2.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_bunny()
    mesh = load_mesh()

    print("Rendering point cloud...")
    render_pointcloud(mesh)

    print("Rendering mesh with wireframe...")
    render_mesh(mesh)

    print("Rendering voxel grid...")
    render_voxel(mesh)

    print("Rendering octree...")
    render_octree(mesh)

    print("Rendering SDF...")
    render_sdf(mesh)

    print(f"\nAll renders saved to ./{OUTPUT_DIR}/")
