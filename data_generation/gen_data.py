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
DATA_DIR = "/home/kln/sandbox/cmu/repos/3d_project/dataset/keypointnet/"
# model_file = "03001627/8d458ab12073c371caa2c06fded3ca21.ply"
# ply_file = os.path.join(DATA_DIR, model_file)


def load_keypoints(kpts_file):
    model_file, kpts_xyz, kpts_rgb = None, None, None
    with open(kpts_file) as f:
        kpts_dict = json.load(f)[0]  # load only first keypoint

        # model_file = os.path.join(kpts_dict["class_id"], kpts_dict["model_id"] + ".ply")
        model_file = os.path.join("02691156/5515a62182afd357f2b0736dd4d8afe0.ply")
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


# kpts_file = os.path.join(DATA_DIR, "chair.json")
kpts_file = os.path.join(DATA_DIR, "airplane.json")
model_file, kpts_xyz, kpts_rgb, kpts_faces = load_keypoints(kpts_file)
ply_file = os.path.join(DATA_DIR, "ShapeNetCore.v2.ply", model_file)

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
# mesh = load_objs_as_meshes([obj_file], device=device)
mesh = IO().load_mesh(ply_file, device=device)

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
sphere_mesh.scale_verts_(0.01)

# textures_rgb = torch.ones_like(verts) * torch.tensor([255, 255, 255]).to(device)
# mesh_faces = mesh.faces_packed()

# for kpt_face in kpts_faces:
#     v = mesh_faces[kpt_face]
#     for i in v:
#         textures_rgb[i] = torch.tensor([255, 0, 0]).to(device)

# textures = TexturesVertex(textures_rgb.unsqueeze(0))
# mesh = Meshes(
#     verts=verts.unsqueeze(0), faces=mesh_faces.unsqueeze(0), textures=textures
# )


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


def generate(elev, azim, batch_num):

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
        kpts_features = torch.ones_like(kpts_xyz) * torch.tensor([255, 255, 255]).to(
            device
        )
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

        batch_size = elev.shape[0]
        kpts_xyz_batch = kpts_xyz.unsqueeze(0).repeat(batch_size, 1, 1)
        # kpts_screen = cameras.transform_points_screen(
        #     kpts_xyz_batch, image_size=(image_size, image_size)
        # )
        # kpts_screen = kpts_screen.cpu().numpy()
        # kpts_rgb = kpts_rgb.cpu().numpy()

        # Custome Shader class to obtain object mask
        class MaskShader(torch.nn.Module):
            def __init__(self, device="cpu"):
                super().__init__()

            def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
                N, H, W, K = fragments.pix_to_face.shape
                device = fragments.pix_to_face.device

                # kpts_masks = torch.zeros(N, H, W, 1)
                # n, h, w = torch.where(fragments.pix_to_face[..., 0] == 20521)
                # kpts_masks[n, h, w, 0] = 1
                # background mask
                # is_background = fragments.pix_to_face[..., 0] < 0
                # is_background = is_background.unsqueeze(3)
                kpts_layers = []
                # for kpt_face, kpt_rgb in zip(kpts_faces, kpts_rgb):
                for i, kpt_face in enumerate(kpts_faces):
                    kpt_fg = fragments.pix_to_face[..., 0] == kpt_face
                    kpt_layer = torch.zeros(N, H, W).to(device) + kpt_fg * (
                        torch.ones(N, H, W).to(device) * (i + 1)
                    )
                    kpts_layers.append(kpt_layer)

                kpts_layers = torch.stack(kpts_layers)
                kpts_masks = torch.max(kpts_layers, dim=0).values

                return kpts_masks  # (N, H, W, 1) RGBA image
                # return kpts_masks.squeeze(3)  # (N, H, W, 1) RGBA image

        # Create a Phong renderer by composing a rasterizer and a shader. The textured
        # Phong shader will interpolate the texture uv coordinates for each vertex,
        # sample from a texture image and apply the Phong lighting model

        blend_params = BlendParams(background_color=(0, 0, 0))
        mesh_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=image_size, blur_radius=0.0, faces_per_pixel=1
                )
            ),
            shader=SoftPhongShader(
                device=device, lights=lights, blend_params=blend_params
            ),
        )
        mask_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=image_size, blur_radius=0.0, faces_per_pixel=1
                )
            ),
            shader=MaskShader(),
        )

        sphere_verts = sphere_mesh.verts_padded()
        texture_rgb = torch.ones_like(sphere_verts) * torch.tensor([255, 0, 0]).to(
            device
        )
        texture = TexturesVertex(texture_rgb)
        text_sphere_mesh = Meshes(sphere_verts, sphere_mesh.faces_padded(), texture)
        spheres = text_sphere_mesh.extend(kpts_xyz.shape[0])
        transform = Transform3d(device=device).compose(
            Translate(kpts_xyz, device=device)
        )
        spheres = spheres.update_padded(
            transform.transform_points(spheres.verts_padded())
        )

        # Create a batch of meshes by repeating the mesh and associated textures.
        # Meshes has a useful `extend` method which allows us do this very easily.
        # This also extends the textures.
        # kpt_mesh = join_meshes_as_scene([mesh, spheres])
        # meshes = kpt_mesh.extend(elev.shape[0])
        meshes = mesh.extend(elev.shape[0])

        # Render the mesh from each viewing angle
        target_images = mesh_renderer(meshes, cameras=cameras, lights=lights)
        kpts_masks = mask_renderer(
            meshes,
            cameras=cameras,
            lights=lights,
            kpts_faces=kpts_faces,
            kpts_rgb=kpts_rgb,
        )

        # generate dataset images
        num_kpts = kpts_faces.shape[0]
        for i, (img, mask) in tqdm(
            enumerate(zip(target_images.cpu().numpy(), kpts_masks.cpu().numpy()))
        ):
            img = (img[..., :3] * 255).astype(np.uint8)
            mask = (mask[:, :, None] * np.array([255, 0, 0]) / num_kpts).astype(
                np.uint8
            )
            fg = mask > 0
            # img = img * ~fg + mask
            # kpts = kpts_screen[i].round().astype(int)
            # for j, kpt in enumerate(kpts):
            #     # img[kpt[1], kpt[0]] = kpts_rgb[j]
            #     r, c = kpt[1], kpt[0]
            #     img[r - 2 : r + 2, c - 2 : c + 2, :] = np.array([255, 0, 0])
            #     print(r, c)
            rgb_img = Image.fromarray(img)
            # mask_img = Image.fromarray(mask)
            idx = batch_num * elev.shape[0] + i
            rgb_img.save(f"./output/image_{idx:06d}.png")
            # mask_img.save(f"./output/mask_{i:06d}.png")

    # gen_kpts_images()
    gen_obj_images()


batch_size = 10
batch_start = 0
step = 0
for i in range(num_samples // batch_size):
    start = batch_start
    end = batch_start + batch_size
    generate(elev[start:end], azim[start:end], i)
    batch_start = end
    step = step + i

if batch_start < num_samples:
    generate(elev[batch_start:], azim[batch_start:], step)

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
