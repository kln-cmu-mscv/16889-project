import sys
import os
import numpy as np
from sklearn import model_selection
import torch
from tqdm.auto import tqdm
import json

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
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
# --------- obj + mtl filetype ------------#
# DATA_DIR = "./data/"
# obj_file = os.path.join(DATA_DIR, "cow.obj")

# ---------- ply filetype ----------------#
DATA_DIR = "/home/kln/sandbox/cmu/repos/3d_project/dataset/keypointnet/"
# model_file = "03001627/8d458ab12073c371caa2c06fded3ca21.ply"
# ply_file = os.path.join(DATA_DIR, model_file)


def load_keypoints(kpts_file):
    model_file, kpts_xyz, kpts_rgb = None, None, None
    with open(kpts_file) as f:
        kpts_dict = json.load(f)[0]  # load only first keypoint

        model_file = os.path.join(kpts_dict["class_id"], kpts_dict["model_id"] + ".ply")
        kpts_xyz = []
        kpts_rgb = []
        for kpt in kpts_dict["keypoints"]:
            kpts_xyz.append(kpt["xyz"])
            kpts_rgb.append(kpt["rgb"])
        kpts_xyz = torch.tensor(kpts_xyz)
        kpts_rgb = torch.tensor(kpts_rgb)

    return model_file, kpts_xyz, kpts_rgb


kpts_file = os.path.join(DATA_DIR, "chair.json")
model_file, kpts_xyz, kpts_rgb = load_keypoints(kpts_file)
ply_file = os.path.join(DATA_DIR, "ShapeNetCore.v2.ply", model_file)

# kpts_xyz = kpts_xyz * torch.tensor([-1, 1, -1])  # transform axis to pytorch3d
kpts_xyz, kpts_rgb = kpts_xyz.to(device), kpts_rgb.to(device)

# kpts_xyz = torch.tensor(
#     [
#         [0.16172105225242855, 0.39077869068739557, 0.1352145785287725],
#         [-0.15787448163662027, 0.40384870078270607, 0.1307143438565415],
#     ]
# ).to(device)

# load mesh obj
# mesh = load_objs_as_meshes([obj_file], device=device)
mesh = IO().load_mesh(ply_file, device=device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
# to its original center and scale.  Note that normalizing the target mesh,
# speeds up the optimization but is not necessary!
def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))
    return mesh


# mesh = normalize_mesh(mesh)


def normalize_kpts(kpts_xyz):
    N = kpts_xyz.shape[0]
    center = kpts_xyz.mean(0)
    scale = max((kpts_xyz - center).abs().max(0)[0])
    kpts_xyz = kpts_xyz - center
    kpts_xyz = kpts_xyz * (1.0 / float(scale))
    return kpts_xyz


# kpts_xyz = normalize_kpts(kpts_xyz)

# number of sample outputs to generate
train_samples = 8
test_samples = 2
num_samples = train_samples + test_samples

image_size = 1024

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


def gen_kpts_images():

    # kpts_pcd = Pointclouds(points=[kpts_xyz], features=[(kpts_rgb * 255) / 255.0])
    kpts_features = torch.ones_like(kpts_xyz) * torch.tensor([255, 255, 255]).to(device)
    kpts_pcd = Pointclouds(points=[kpts_xyz], features=[kpts_features])

    kpts_pcd = kpts_pcd.extend(num_samples)

    points_renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=cameras,
            raster_settings=PointsRasterizationSettings(
                image_size=image_size, radius=0.003, points_per_pixel=10
            ),
        ),
        compositor=AlphaCompositor(),
    )

    # generate keypoint images
    kpts_images = points_renderer(kpts_pcd)
    for i, img in tqdm(enumerate(kpts_images.cpu().numpy())):
        img = (img[..., :3] * 255).astype(np.uint8)
        rgb_img = Image.fromarray(img)
        rgb_img.save(f"./output/kpts_{i:06d}.png")


def gen_obj_images():

    global kpts_rgb

    kpts_xyz_batch = kpts_xyz.unsqueeze(0).repeat(num_samples, 1, 1)
    kpts_screen = cameras.transform_points_screen(
        kpts_xyz_batch, image_size=(image_size, image_size)
    )
    kpts_screen = kpts_screen.cpu().numpy()
    kpts_rgb = kpts_rgb.cpu().numpy()

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=image_size, blur_radius=0.0, faces_per_pixel=1
            )
        ),
        shader=SoftPhongShader(device=device, lights=lights),
    )

    # Create a batch of meshes by repeating the mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(num_samples)

    # Render the mesh from each viewing angle
    target_images = mesh_renderer(meshes, cameras=cameras, lights=lights)

    # generate dataset images
    for i, img in tqdm(enumerate(target_images.cpu().numpy())):
        img = (img[..., :3] * 255).astype(np.uint8)
        kpts = kpts_screen[i].round().astype(int)
        for j, kpt in enumerate(kpts):
            # img[kpt[1], kpt[0]] = kpts_rgb[j]
            r, c = kpt[1], kpt[0]
            img[r - 2 : r + 2, c - 2 : c + 2, :] = np.array([255, 0, 0])
            print(r, c)
        rgb_img = Image.fromarray(img)
        rgb_img.save(f"./output/image_{i:06d}.png")


gen_kpts_images()
gen_obj_images()


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
scene = plot_scene({"figure": {"Mesh": mesh, "Camera": cameras}})
scene.show()
