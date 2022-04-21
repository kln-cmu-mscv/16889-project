import torch
import math
import sys
import os
import numpy as np
import torch
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage 
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    HardPhongShader,
    TexturesVertex,
)

from PIL import Image

def Cart_to_Spherical(xyz):
    #takes list xyz (single coord)
    
    x       = xyz[:,0]
    y       = xyz[:,1]
    z       = xyz[:,2]
    r       =  torch.sqrt(x*x + y*y + z*z)
    theta   =  torch.cos(z/r)*180/ math.pi #to degrees
    phi     =  torch.atan2(y,x)*180/ math.pi
    return r, theta, phi

# TODO: change r
def Spherical_to_Cart(theta, phi, r = 3.0):
    x = r * torch.cos(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.cos(theta)
    z = r * torch.sin(theta)
    
    return torch.stack([x,y,z], dim = 1)


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def constraint(output):
    return torch.sum((torch.abs(torch.cos(output[0]) **2 + torch.sin(output[0]) **2 - 1))**2)

def render_image(theta, phi):
    # setup cuda/cpu device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device=device)
        # clear cache
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    # setup data
    DATA_DIR = "./dataset/Dataset/"
    model_file = "03001627/8d458ab12073c371caa2c06fded3ca21"
    obj_file = os.path.join(DATA_DIR, "ShapeNetCore.v2", model_file, "models/model_normalized.obj"
)
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
    # change obj mesh textures to rainbow colors
    verts = mesh.verts_padded()
    faces = mesh.faces_padded()
    texture_rgb = (verts - verts.min()) / (verts.max() - verts.min())
    texture = TexturesVertex(texture_rgb)
    mesh = Meshes(verts, faces, texture)

    image_size = 256

    cam_dist = 3.0
    
    elev = theta.squeeze()
    azim = phi.squeeze()

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
        shader=HardPhongShader(device=device, 
                               lights=lights, 
                               cameras=cameras, 
                               blend_params=BlendParams(background_color=(0,0,0))
        ),
    )

    # Create a batch of meshes by repeating the mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(len(theta))

    # Render the mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # generate dataset images
    rendered_images = []
    for i, img in tqdm(enumerate(target_images)):

        rgb_img = (img[..., :3]).to(torch.float)
        rgb_img= torch.permute(rgb_img, (2,0,1))
        rendered_images.append(rgb_img)
        
    rendered_images = torch.stack(rendered_images, dim=0)
    
    return rendered_images.to('cuda')



