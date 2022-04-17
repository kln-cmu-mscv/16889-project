import sys
import os
import numpy as np
import torch
from tqdm.auto import tqdm

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
)
from pytorch3d.vis.plotly_vis import plot_scene
from PIL import Image

# setup cuda/cpu device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device=device)
    # clear cache
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

# setup data
DATA_DIR = "./data/"
obj_file = os.path.join(DATA_DIR, "cow.obj")

# load mesh obj
mesh = load_objs_as_meshes([obj_file], device=device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
# to its original center and scale.  Note that normalizing the target mesh,
# speeds up the optimization but is not necessary!
verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))

# number of sample outputs to generate
train_samples = 80
test_samples = 20
num_samples = train_samples + test_samples

image_size = 256

cam_dist = 3.0
# Get a batch of viewing angles.
elev = torch.randint(0, 360, (num_samples, 1)).squeeze()
azim = torch.randint(-180, 180, (num_samples, 1)).squeeze()
# elev = torch.linspace(0, 360, num_samples)
# azim = torch.linspace(-180, 180, num_samples)

# Place a point light in front of the object. As mentioned above, the front of
# the object is facing the -z direction.
lights = PointLights(device=device, location=[[0.0, 0.0, -cam_dist]])

# Initialize an OpenGL perspective camera that represents a batch of different
# viewing angles. All the cameras helper methods support mixed type inputs and
# broadcasting. So we can view the camera from the a distance of dist=2.7, and
# then specify elevation and azimuth angles for each viewpoint as tensors.
R, T = look_at_view_transform(dist=cam_dist, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Create a Phong renderer by composing a rasterizer and a shader. The textured
# Phong shader will interpolate the texture uv coordinates for each vertex,
# sample from a texture image and apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        raster_settings=RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
    ),
    shader=SoftPhongShader(device=device, lights=lights),
)

# Create a batch of meshes by repeating the mesh and associated textures.
# Meshes has a useful `extend` method which allows us do this very easily.
# This also extends the textures.
meshes = mesh.extend(num_samples)

# Render the mesh from each viewing angle
target_images = renderer(meshes, cameras=cameras, lights=lights)

# generate dataset images
for i, img in tqdm(enumerate(target_images.cpu().numpy())):
    rgb_img = Image.fromarray((img[..., :3] * 255).astype(np.uint8))
    rgb_img.save(f"./output/image_{i:06d}.png")

# save pose information for dataset
pose = torch.stack([elev, azim, torch.ones(num_samples) * cam_dist]).cpu().numpy().T
np.save("./output/pose.npy", pose)

# generate train-test split for dataset
idx = np.random.permutation(num_samples)
train_idx = idx[:train_samples]
test_idx = idx[-test_samples:]
np.save("./output/train_indices.npy", train_idx)
np.save("./output/test_indices.npy", test_idx)

# DEBUG: show scene rendering for debugging
# scene = plot_scene({"figure": {"Mesh": mesh, "Camera": cameras}})
# scene.show()
