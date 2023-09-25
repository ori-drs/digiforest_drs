import open3d as o3d
import numpy as np


def bbox_to_mesh(
    bbox, width=None, height=None, depth=None, offset=[0, 0, 0], color=[0, 0, 0]
):
    w = bbox.get_extent()[0].item() if width is None else width
    h = bbox.get_extent()[1].item() if height is None else height
    d = bbox.get_extent()[2].item() if depth is None else depth

    mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
    position = (bbox.get_center() - bbox.get_extent() / 2).numpy() + np.array(offset)
    mesh.translate(position)
    mesh.paint_uniform_color(color)

    return mesh
