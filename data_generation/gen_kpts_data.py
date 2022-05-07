import sys
import os
import numpy as np
from sklearn import model_selection
import torch
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
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
    HardPhongShader,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene
from PIL import Image
from pytorch3d.utils import ico_sphere
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.transforms.transform3d import Transform3d, Rotate, Translate, Scale
from pytorch3d.structures import join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.renderer.blending import BlendParams

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
DATA_DIR = "/home/kln/sandbox/cmu/repos/3d_project/dataset/"
# model_file = "03001627/8d458ab12073c371caa2c06fded3ca21.ply"
# ply_file = os.path.join(DATA_DIR, model_file)


def load_keypoints(kpts_file):
    model_file, kpts_xyz, kpts_rgb = None, None, None
    with open(kpts_file) as f:
        kpts_dict = json.load(f)[0]  # load only first keypoints

        model_file = os.path.join(kpts_dict["class_id"], kpts_dict["model_id"])
        kpts_xyz = []
        kpts_rgb = []
        kpts_faces = []
        for kpt in kpts_dict["keypoints"]:
            kpts_xyz.append(kpt["xyz"])
            kpts_rgb.append(kpt["rgb"])
            kpts_faces.append(kpt["mesh_info"]["face_index"])
        kpts_xyz = torch.tensor(kpts_xyz)
        kpts_rgb = torch.tensor(kpts_rgb)
        kpts_faces = torch.tensor(kpts_faces)

    return model_file, kpts_xyz, kpts_rgb, kpts_faces


kpts_file = os.path.join(DATA_DIR, "keypointnet/chair.json")
# kpts_file = os.path.join(DATA_DIR, "airplane.json")
model_file, kpts_xyz, kpts_rgb, kpts_faces = load_keypoints(kpts_file)
ply_file = os.path.join(
    DATA_DIR, "keypointnet/ShapeNetCore.v2.ply", model_file + ".ply"
)
obj_file = os.path.join(
    DATA_DIR, "ShapeNetCore.v2", model_file, "models/model_normalized.obj"
)

# kpts_xyz = kpts_xyz * torch.tensor([-1, 1, -1])  # transform axis to pytorch3d
kpts_xyz, kpts_rgb, kpts_faces = (
    kpts_xyz.to(device),
    kpts_rgb.to(device),
    kpts_faces.to(device),
)

# kpts_xyz = torch.tensor(
#     [
#         [0.16172105225242855, 0.39077869068739557, 0.1352145785287725],
#         [-0.15787448163662027, 0.40384870078270607, 0.1307143438565415],
#     ]
# ).to(device)

# load mesh obj
mesh = load_objs_as_meshes([obj_file], device=device)
# mesh = IO().load_mesh(ply_file, device=device)

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
obj_mesh = mesh.clone()
# nomalize keypoints also
scale = max((kpts_xyz - center).abs().max(0)[0])
kpts_xyz = kpts_xyz - center
kpts_xyz = kpts_xyz * (1.0 / float(scale))

sphere_mesh = ico_sphere(4, device)
verts = sphere_mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
sphere_mesh.offset_verts_(-center)
sphere_mesh.scale_verts_((1.0 / float(scale)))
sphere_mesh.scale_verts_(0.009)

# change obj mesh textures to rainbow colors
verts = mesh.verts_padded()
faces = mesh.faces_padded()
texture_rgb = (verts - verts.min()) / (verts.max() - verts.min())
texture = TexturesVertex(texture_rgb)
obj_mesh = Meshes(verts, faces, texture)
del mesh


# number of sample outputs to generate
train_samples = 800
test_samples = 200
num_samples = train_samples + test_samples

image_size = 1024

cam_dist = 3.0
# Get a batch of viewing angles.
# elev = torch.randint(0, 360, (num_samples, 1)).squeeze()
# azim = torch.randint(-180, 180, (num_samples, 1)).squeeze()
elev = torch.linspace(-90, 90)
azim = torch.linspace(-180, 180)

grid_x, grid_y = torch.meshgrid(elev, azim, indexing="ij")
grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
idx = torch.randperm(grid_x.shape[0])[:num_samples]
elev = grid_x[idx]
azim = grid_y[idx]

# custome Renderer to get depth
class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def generate(elev, azim):
    lights = PointLights(device=device, location=[[0.0, 0.0, -cam_dist]])
    R, T = look_at_view_transform(dist=cam_dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    image_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=image_size, blur_radius=0.0, faces_per_pixel=1
            )
        ),
        shader=SoftPhongShader(
            device=device,
            lights=lights,
        ),
    )

    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=image_size, blur_radius=0.0, faces_per_pixel=1
            )
        ),
        shader=HardPhongShader(
            device=device,
            lights=lights,
            cameras=cameras,
            blend_params=BlendParams(background_color=(0, 0, 0)),
        ),
    )

    # rendered_images = image_renderer(join_meshes_as_scene([obj_mesh]), cameras=cameras)
    # obj_img = rendered_images[0].cpu().numpy()

    sphere_verts = sphere_mesh.verts_padded()
    texture_rgb = torch.ones_like(sphere_verts) * torch.tensor([255, 0, 0]).to(device)
    texture = TexturesVertex(texture_rgb)
    text_sphere_mesh = Meshes(sphere_verts, sphere_mesh.faces_padded(), texture)
    spheres = text_sphere_mesh.extend(kpts_xyz.shape[0])
    transform = Transform3d(device=device).compose(Translate(kpts_xyz, device=device))
    spheres = spheres.update_padded(transform.transform_points(spheres.verts_padded()))

    verts = obj_mesh.verts_padded()
    faces = obj_mesh.faces_padded()
    texture_rgb = torch.ones_like(verts) * torch.tensor([0, 255, 0]).to(device)
    texture = TexturesVertex(texture_rgb)
    chair_mesh = Meshes(verts, faces, texture)

    rendered_images, rendered_depth = renderer(
        join_meshes_as_batch([obj_mesh, spheres]), cameras=cameras
    )
    obj_img = rendered_images[0].cpu().numpy()[..., :3]
    obj_mask = rendered_images[0].cpu().numpy()[..., 3]

    depth_maps = rendered_depth.cpu().numpy()
    chair_depth = depth_maps[0]

    kpts_layers = []
    map = {}
    for i, kpt_depth in enumerate(depth_maps[1:]):
        mask = kpt_depth > 0
        d = kpt_depth.max().round() * (i + 1)
        map[d] = i + 1
        # print(i, d, mask.sum())
        fg = kpt_depth < chair_depth
        img = np.zeros_like(fg) + (np.ones_like(fg) * d) * fg
        img = img * mask
        kpts_layers.append(img)

    kpts_layers = np.stack(kpts_layers)
    kpts = np.max(kpts_layers, axis=0)

    for d, i in map.items():
        x = np.where(kpts == d)
        kpts[x] = i

    rendered_images = image_renderer(
        join_meshes_as_scene([chair_mesh, spheres]), cameras=cameras
    )
    combined_img = rendered_images[0].cpu().numpy()

    return kpts, obj_img, combined_img, obj_mask


for i, (theta, phi) in tqdm(enumerate(zip(elev, azim)), total=elev.shape[0]):
    kpts, obj, combined, obj_mask = generate(theta, phi)
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))
    for ax, im in zip(
        axs.ravel(), [kpts, obj[..., :3], combined[..., :3] / 255, obj_mask]
    ):
        ax.imshow(im)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"./output/viz_{i:06d}.jpg")
    plt.close(fig)
    # kpts = kpts.astype(np.uint8) * np.ones(3, dtype=np.uint8)[None, None, :]
    kpts = kpts.squeeze(2).astype(np.uint8)
    kpts_img = Image.fromarray(kpts)
    obj_img = Image.fromarray((obj[..., :3] * 255).astype(np.uint8))
    obj_mask_img = Image.fromarray((obj_mask * 255).astype(np.uint8))
    combined_img = Image.fromarray((combined[..., :3]).astype(np.uint8))
    kpts_img.save(f"./output/kpts_{i:06d}.png")
    obj_img.save(f"./output/image_{i:06d}.png")
    obj_mask_img.save(f"./output/mask_{i:06d}.png")
    combined_img.save(f"./output/combined_{i:06d}.png")

# save pose information for dataset
pose = torch.stack([elev, azim, torch.ones(num_samples) * cam_dist]).cpu().numpy().T
np.save("./output/pose.npy", pose)

# generate train-test split for dataset
idx = np.random.permutation(num_samples)
train_idx = idx[:train_samples]
test_idx = idx[-test_samples:]
np.save("./output/train_indices.npy", train_idx)
np.save("./output/test_indices.npy", test_idx)
