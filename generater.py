#https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9
"""from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

your_mesh = mesh.Mesh.from_file('3DBenchy.stl')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

pyplot.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import trimesh

from pyrender import (OffscreenRenderer, PerspectiveCamera, DirectionalLight, SpotLight, Mesh, Node, Scene)

fuze_trimesh = trimesh.load('3DBenchy.stl')
fuze_mesh = Mesh.from_trimesh(fuze_trimesh)


# Instanced
poses = np.tile(np.eye(4), (2, 1, 1))
poses[0, :3, 3] = np.array([-0.1, -0.10, 0.05])
poses[1, :3, 3] = np.array([-0.15, -0.10, 0.05])

direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)

cam = PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.array([
    [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0.5],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0.4],
    [0.0, 0.0, 0.0, 1.0]
])

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02]))

fuze_node = Node(mesh=fuze_mesh, translation=np.array([
    0.1, .15, -np.min(fuze_trimesh.vertices[:, 2])
]))
scene.add_node(fuze_node)

#_ = scene.add(direc_l, pose=cam_pose)
#_ = scene.add(spot_l, pose=cam_pose)

_ = scene.add(cam, pose=cam_pose)

r = OffscreenRenderer(viewport_width=640, viewport_height=480)
color, depth = r.render(scene)

imgplot = plt.imshow(color)
#img = plt.imshow(depth)
plt.show()"""