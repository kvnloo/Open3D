# Copyright (c) 2018-2023 Open3D contributors
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import open3d as o3d


def apply_color(pcd,
                octree,
                mode,
                cmap_name=None,
                z_normalized=None,
                rgb_colors=None):
    """Color points and their octree leaves with the same source colors."""
    if mode == 'height':
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name)
        colors = cmap(z_normalized)[:, :3]
    else:
        colors = rgb_colors

    pcd.colors = o3d.utility.Vector3dVector(colors)
    point_colors = np.asarray(pcd.colors)

    def traverse_and_color(node, node_info):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            # Leaf indices reflect half-open octree insertion boundaries.
            # A full-cloud bounds mask would include neighboring leaf points.
            indices = node.indices
            if indices:
                node.color = np.mean(
                    point_colors[np.asarray(indices, dtype=np.intp)], axis=0)
            else:
                node.color = np.array([0.5, 0.5, 0.5])
        return False

    octree.traverse(traverse_and_color)
    return pcd, octree


def main():
    """Create a sample octree and allow interactive height/RGB coloring."""
    xyz = np.random.rand(10000, 3) * 10
    rgb_colors = np.random.rand(10000, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    z = xyz[:, 2]
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
              center=pcd.get_center())

    print('Octree division')
    octree = o3d.geometry.Octree(max_depth=5)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    color_maps = ['jet', 'hot', 'viridis', 'cool']
    current_cmap_index = 0
    color_mode = 'height'
    pcd, octree = apply_color(pcd,
                              octree,
                              'height',
                              cmap_name=color_maps[current_cmap_index],
                              z_normalized=z_normalized,
                              rgb_colors=rgb_colors)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(
            window_name='Octree and Point Cloud Visualization'):
        raise RuntimeError('Could not create Open3D visualization window')
    vis.add_geometry(pcd)
    vis.add_geometry(octree)

    def toggle_color_map(vis):
        nonlocal current_cmap_index, pcd, octree
        if color_mode == 'height':
            current_cmap_index = (current_cmap_index + 1) % len(color_maps)
            new_cmap = color_maps[current_cmap_index]
            print(f"Switching to color map: {new_cmap} (height mode)")
            pcd, octree = apply_color(pcd,
                                      octree,
                                      'height',
                                      cmap_name=new_cmap,
                                      z_normalized=z_normalized,
                                      rgb_colors=rgb_colors)
            vis.update_geometry(pcd)
            vis.update_geometry(octree)
            vis.update_renderer()
        else:
            print("In RGB mode; press '5' to switch to height-based coloring.")
        return False

    def toggle_color_mode(vis):
        nonlocal color_mode, pcd, octree
        color_mode = 'rgb' if color_mode == 'height' else 'height'
        if color_mode == 'height':
            print("Switching to height mode with color map: "
                  f"{color_maps[current_cmap_index]}")
            pcd, octree = apply_color(
                pcd,
                octree,
                'height',
                cmap_name=color_maps[current_cmap_index],
                z_normalized=z_normalized,
                rgb_colors=rgb_colors)
        else:
            print("Switching to RGB mode")
            pcd, octree = apply_color(pcd,
                                      octree,
                                      'rgb',
                                      rgb_colors=rgb_colors)
        vis.update_geometry(pcd)
        vis.update_geometry(octree)
        vis.update_renderer()
        return False

    vis.register_key_callback(ord('4'), toggle_color_map)
    vis.register_key_callback(ord('5'), toggle_color_mode)
    print("Visualizing with initial color map: jet (height mode)")
    print("Press '4' to cycle color maps "
          "(jet, hot, viridis, cool) in height mode")
    print("Press '5' to toggle between height-based and RGB-based coloring")
    try:
        vis.run()
    finally:
        vis.destroy_window()


if __name__ == '__main__':
    main()
