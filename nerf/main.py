import os
import warnings

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import imageio
import wandb

from keypoints import KeyPoints
from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import matplotlib.pyplot as plt

from implicit import volume_dict
from sampler import sampler_dict
from renderer import renderer_dict
from ray_utils import (
    sample_images_at_xy,
    sample_keypoints_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels
)
from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    vis_grid,
    vis_rays,
)
from dataset import (
    get_nerf_datasets,
    trivial_collate,
)
from render_functions import (
    render_points
)


# Model class containing:
#   1) Implicit volume defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit volume given a sampling scheme

class Model(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.train_keypoints = cfg.train_keypoints

        # Get implicit function from config
        self.implicit_fn = volume_dict[cfg.implicit_function.type](
            cfg.implicit_function,
            self.train_keypoints
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](
            cfg.sampler
        )

        # Initialize volume renderer
        self.renderer = renderer_dict[cfg.renderer.type](
            cfg.renderer,
            self.train_keypoints
        )

    def forward(
        self,
        ray_bundle
    ):
        # Call renderer with
        #  a) Implicit volume
        #  b) Sampling routine

        return self.renderer(
            self.sampler,
            self.implicit_fn,
            ray_bundle
        )

def render_images(
    model,
    cameras,
    image_size,
    save=False,
    file_prefix='',
    train_keypoints=False
):
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')

        torch.cuda.empty_cache()
        camera = camera.to(device)
        xy_grid = get_pixels_from_image(image_size, camera) # TODO (1.3): implement in ray_utils.py
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera) # TODO (1.3): implement in ray_utils.py
        ray_bundle = model.sampler(ray_bundle)

        out = model(ray_bundle)

        if train_keypoints:
            keypoints = torch.argmax(F.softmax(out['keypoints'].view(image_size[1],
                                                                     image_size[0],
                                                                     -1),
                                               dim=-1),
                                     dim=-1).detach().cpu().numpy()
            ignore_label = KeyPoints.KEYPOINTS_NAME_TO_I['not-keypoint']
            keypoints = np.argwhere(keypoints != ignore_label)

            # keypoints_x = keypoints[:, 1]
            # keypoints_y = keypoints[:, 0]
            # keypoints_colors =

        # Return rendered features (colors)
        image = np.array(
            out['feature'].view(
                image_size[1], image_size[0], 3
            ).detach().cpu()
        )

        if train_keypoints:

            neighbors = [(-1, 0),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1)]

            for i in range(keypoints.shape[0]):
                # For now same colors.
                image[keypoints[i][0], keypoints[i][1]] = np.array([0.22352941,
                                                                    1.,
                                                                    0.07843137])
                for dx, dy in neighbors:

                    if int(keypoints[i][0] + dy) >= image_size[1] or \
                       int(keypoints[i][1] + dx) >= image_size[0]:
                        continue

                    image[keypoints[i][0] + dy, keypoints[i][1] + dx] = np.array([0.22352941,
                                                                                  1.,
                                                                                  0.07843137])

        all_images.append(image)

        # TODO (1.5): Visualize depth
        if cam_idx == 2 and file_prefix == '':
            plt.imsave('results/depth1.png',
                       np.array(out['depth'].view(image_size[1],
                                                  image_size[0]).detach().cpu()))

        # Save
        if save:
            plt.imsave(
                f'{file_prefix}_{cam_idx}.png',
                image
            )

    return all_images

def create_model(cfg):
    # Create model
    model = Model(cfg)
    model.cuda(); model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path

def train_nerf(
    cfg
):
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)

    if cfg.logging.use_wandb:
        wandb.watch(model, log_freq=1000)

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
        train_keypoints = cfg.train_keypoints
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    criterion = torch.nn.MSELoss()

    if cfg.train_keypoints:
        weights = torch.ones(cfg.implicit_function.n_keypoints).cuda()
        weights[KeyPoints.KEYPOINTS_NAME_TO_I['not-keypoint']] /= cfg.implicit_function.n_keypoints
        keypoints_crit = torch.nn.CrossEntropyLoss(weight=weights)
        # keypoints_crit = torch.nn.CrossEntropyLoss()

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            )
            ray_bundle = get_rays_from_pixels(
                xy_grid, cfg.data.image_size, camera
            )

            rgb_gt = sample_images_at_xy(image[..., :3], xy_grid)

            # Run model forward
            out = model(ray_bundle.to(torch.device('cuda')))

            recon_loss = criterion(out['feature'].view(rgb_gt.shape), rgb_gt)

            keypoints_loss = 0.
            keypoints_acc = 0.

            if cfg.train_keypoints:
                pred_keypoints = out['keypoints']
                label_keypoints = \
                    sample_keypoints_at_xy(image[..., -1].view(-1,
                                                               image.size(1),
                                                               image.size(2),
                                                               1),
                                           xy_grid)
                
                # (1024, C), (1024)
                keypoints_loss = keypoints_crit(pred_keypoints,
                                                label_keypoints.view(-1).long())
                pred_labels = torch.argmax(torch.softmax(pred_keypoints, dim=-1),
                                           dim=-1)
                pred_labels = pred_labels[pred_labels != KeyPoints.KEYPOINTS_NAME_TO_I['not-keypoint']]
                label_keypoints = label_keypoints.view(-1)[label_keypoints.view(-1) != KeyPoints.KEYPOINTS_NAME_TO_I['not-keypoint']]
                keypoints_acc = torch.sum(pred_labels == label_keypoints) / \
                    pred_labels.size(0)

            loss = recon_loss + 10 * keypoints_loss

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, ' +
                                    f'Loss: {loss:.06f}, ' +
                                    f'Recon Loss: {recon_loss:.06f}, ' +
                                    f'Keypoints Loss: {keypoints_loss:.06f}'
                                    f'Keypoints Acc: {keypoints_acc:.06f}')
            t_range.refresh()

            if cfg.logging.use_wandb:
                wandb.log({'train/step': epoch * len(train_dataloader) + iteration})
                wandb.log({'train/recon_loss': recon_loss.item()})
                wandb.log({'train/LR': optimizer.param_groups[0]['lr']})

                if cfg.train_keypoints:
                    wandb.log({'train/keypoint_loss': keypoints_loss.item()})
                    wandb.log({'train/keypoint_acc': keypoints_acc.item()})

            del ray_bundle
            del out

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            (epoch + 1) % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (epoch + 1) % cfg.training.render_interval == 0:
            with torch.no_grad():
                test_images = render_images(
                    model, 
                    create_surround_cameras(4.0,
                                            n_poses=20,
                                            up=(0.0, 0.0, 1.0),
                                            focal_length=2.0),
                    cfg.data.image_size,
                    file_prefix='nerf',
                    train_keypoints=cfg.train_keypoints,
                    log_wandb=(cfg.logging.use_wandb and (epoch + 1) % \
                        cfg.logging.render_interval)
                )
                imageio.mimsave(f'results/exp4/nerf_{epoch}.gif',
                                [np.uint8(im * 255) for im in test_images])


@hydra.main(config_path='./configs', config_name='nerf_lego')
def main(cfg: DictConfig):

    os.chdir(hydra.utils.get_original_cwd())

    if cfg.logging.use_wandb:
        wandb.init(project='16889-project-kpnerf')
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

    train_nerf(cfg)

if __name__ == "__main__":
    main()

