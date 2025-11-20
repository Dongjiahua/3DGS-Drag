import os
import sys
import json
import random
from dataclasses import dataclass
from random import randint

import numpy as np
import torch
import torchvision
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene
from scene.gaussian_model_drag import GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.perceptual import PerceptualLoss
from utils.train_dreambooth import DreamBooth, resize_shape

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class Drag:
    def __init__(self, gaussians: GaussianModel, drag_cfg_path: str, steps: int = 10):
        with open(drag_cfg_path, "r") as f:
            drag_cfg = json.load(f)

        self.gaussians = gaussians
        self.drag_cfg = drag_cfg
        self.steps = steps

        self.startpoints = torch.tensor(drag_cfg["start"], device="cuda")
        self.direction = torch.tensor(drag_cfg["direction"], device="cuda")

        self.targets = self.startpoints + self.direction
        self.cur_step = 0
        self.track_mask = None
        self.start_cluster = []

        # mode is a bool in drag config, default False if not provided
        self.mode = drag_cfg["mode"]

        with torch.no_grad():
            self.setup_drag_idx(drag_cfg["voxel_size"])

    def drag_loss(self) -> torch.Tensor:
        distances = 0
        for i, _ in enumerate(self.startpoints):
            drag_idx = i + 1
            target_point = self.targets[i]
            l2_loss = torch.norm(
                self.gaussians._xyz[self.gaussians._drag_idx == drag_idx]
                - target_point.unsqueeze(0),
                dim=1,
            )
            distances += l2_loss.mean()
        return distances

    def track_point(self):
        distances = []
        for i, point in enumerate(self.startpoints):
            average_distance = torch.mean(
                self.gaussians._xyz[self.track_mask[i]] - self.start_cluster[i],
                dim=0,
            )
            new_start_point = point + average_distance
            self.startpoints[i] = new_start_point
            distances.append(
                torch.norm(self.targets[i] - self.startpoints[i]).cpu().item()
            )
        return distances

    def render_all_drag_mask(self, view_stack, pipe, background):
        mask = (self.gaussians._drag_idx > 0)[..., None].float().repeat(1, 3)
        res = []
        for view in view_stack:
            semantic_map = render(
                view,
                self.gaussians,
                pipe,
                background,
                override_color=mask,
            )["render"].detach()
            semantic_map = torch.norm(semantic_map, dim=0)
            semantic_map = torch.nn.functional.max_pool2d(
                semantic_map[None, None].float(),
                kernel_size=21,
                stride=1,
                padding=10,
            ).squeeze()
            semantic_map = semantic_map > 0.8
            res.append(semantic_map.cpu())
        return res

    def track_point_idx(self):
        distances = []
        for i, point in enumerate(self.startpoints):
            drag_idx = i + 1
            average_distance = (
                self.gaussians._xyz[self.gaussians._drag_idx == drag_idx].mean(dim=0)
                - point
            )
            new_start_point = point + average_distance
            self.startpoints[i] = new_start_point
            distances.append(
                torch.norm(self.targets[i] - self.startpoints[i]).cpu().item()
            )
        return distances

    def setup_drag_idx(self, voxel_size: float):
        self.gaussians.init_drag_idx()
        mask = None
        for i, _ in enumerate(self.startpoints):
            drag_idx = i + 1
            mask = self.gaussians.set_drag_idx(
                self.startpoints[i], mask, drag_idx, voxel_size
            )
            self.startpoints[i] = torch.mean(
                self.gaussians._xyz[self.gaussians._drag_idx == drag_idx], dim=0
            )

        if "cluster" in self.drag_cfg:
            self.idxs_list = [[] for _ in range(max(self.drag_cfg["cluster"]) + 1)]
            for i, _ in enumerate(self.startpoints):
                drag_idx = i + 1
                self.idxs_list[self.drag_cfg["cluster"][i]].append(drag_idx)
        else:
            self.idxs_list = [range(1, len(self.startpoints) + 1)]
        print(mask.sum().cpu().item(), "points are selected for dragging.")


    def drag_copy_paste_step_deform(self):
        if self.cur_step >= self.steps:
            return
        step_vec = self.direction / float(self.steps)
        start_points = self.startpoints
        target_points = self.startpoints + step_vec
        # updated call: deform_points(source_points, target_points, idxs_list, mode)
        self.gaussians.deform_points(start_points, target_points, self.idxs_list, self.mode)
        self.startpoints = target_points.detach()
        self.cur_step += 1


def render_sets_train(viewpoint_stack, gaussians, pipe, background):
    results = []
    with torch.no_grad():
        for view in tqdm(viewpoint_stack, desc="Rendering progress"):
            rendering = render(view, gaussians, pipe, background)["render"]
            results.append(rendering.detach().cpu())
    return results


@dataclass
class TrainingState:
    gaussians: GaussianModel
    scene: Scene
    original_cameras: list
    original_images: list
    drag: Drag
    sorted_idx: torch.Tensor | None
    ori_segmentions: list
    oo_segmentions: list
    p_ssim: PerceptualLoss
    model: DreamBooth
    background: torch.Tensor


@dataclass
class DragSchedule:
    drag_steps: int
    anneal_times: int
    start_strength: float
    end_strength: float
    train_iterations_per_edit: int
    guidance_scale: float


def prepare_output_and_logger(args):
    if not getattr(args, "model_path", None):
        args.model_path = "./output"
        os.makedirs(args.model_path, exist_ok=True)

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def init_scene_and_gaussians(dataset, opt, args, pipe) -> tuple[TrainingState, DragSchedule]:
    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # All training cameras (full set)
    all_cameras = scene.getTrainCameras().copy()
    original_cameras = all_cameras.copy()  # used later for "render_real" etc.

    if args.ply is not None:
        gaussians.load_ply(args.ply)

    if args.start_checkpoint:
        model_params, _ = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    drag = Drag(gaussians, args.drag_path, steps=args.drag_steps)

    import time
    time_prefix = time.strftime("%Y%m%d-%H%M%S")
    scene.model_path = os.path.join(
        scene.model_path,
        os.path.splitext(os.path.basename(args.drag_path))[0],
        time_prefix,
    )

    p_ssim = PerceptualLoss().eval().to("cuda")
    model = DreamBooth()

    # --- 1) Masks for ALL cameras ---
    init_seg_all = drag.render_all_drag_mask(
        all_cameras.copy(), pipe, background
    )

    for i, seg in enumerate(init_seg_all):
        original_cameras[i].viewmask = seg
        original_cameras[i].idx = i

    # --- 2) Rank cameras by how much they see the dragged region ---
    semantions = torch.stack(init_seg_all, dim=0).cuda().sum(dim=(1, 2))
    sorted_idx = torch.argsort(semantions, descending=True)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    sorted_idx = sorted_idx[torch.randperm(len(sorted_idx))]

    # indices of chosen training cameras
    chosen_idx = sorted_idx[: args.camera_num]

    # --- 3) Subset both cameras and masks consistently ---
    train_cameras = [all_cameras[i] for i in chosen_idx]
    train_seg = [init_seg_all[i] for i in chosen_idx]

    scene.train_cameras[1.0] = train_cameras

    with torch.no_grad():
        images = render_sets_train(original_cameras, gaussians, pipe, background)
        images = torch.stack(images, dim=0)
        images = resize_shape(images, short=True)

        save_dir = os.path.join(scene.model_path, "origin")
        os.makedirs(save_dir, exist_ok=True)
        for idx, img in enumerate(images):
            torchvision.utils.save_image(img, f"{save_dir}/{idx}.png")

    model.train_dreambooth(save_dir)

    # Now get original_images only for TRAIN cameras (subset)
    original_images = [view.original_image.cpu() for view in scene.getTrainCameras()]

    # Still okay to compute oo_seg on all original_cameras
    oo_seg_orig = drag.render_all_drag_mask(
        original_cameras.copy(), pipe, background
    )

    state = TrainingState(
        gaussians=gaussians,
        scene=scene,
        original_cameras=original_cameras,
        original_images=original_images,
        drag=drag,
        sorted_idx=sorted_idx,
        ori_segmentions=train_seg,      # <- **subset**, aligned with train_cameras
        oo_segmentions=oo_seg_orig,
        p_ssim=p_ssim,
        model=model,
        background=background,
    )

    schedule = DragSchedule(
        drag_steps=args.drag_steps,
        anneal_times=args.anneal_times,
        start_strength=args.start_strength,
        end_strength=args.end_strength,
        train_iterations_per_edit=args.train_iterations_per_edit,
        guidance_scale=args.guidance_scale,
    )

    return state, schedule


def apply_one_drag_step(state: TrainingState, pipe, step_idx: int):
    drag = state.drag
    scene = state.scene
    original_cameras = state.original_cameras
    background = state.background

    with torch.no_grad():
        drag.drag_copy_paste_step_deform()

        step_seg_train = drag.render_all_drag_mask(
            scene.getTrainCameras().copy(), pipe, background
        )
        state.ori_segmentions = [
            (state.ori_segmentions[i] | step_seg_train[i])
            for i in range(len(step_seg_train))
        ]

        step_seg_orig = drag.render_all_drag_mask(
            original_cameras.copy(), pipe, background
        )
        state.oo_segmentions = [
            (state.oo_segmentions[i] | step_seg_orig[i])
            for i in range(len(step_seg_orig))
        ]



        for idx, view in enumerate(original_cameras):
            view.viewmask = state.oo_segmentions[idx]

        for idx, view in enumerate(scene.getTrainCameras()):
            view.viewmask = state.ori_segmentions[idx]
            view.idx = idx


def compute_step_strengths(schedule: DragSchedule):
    if schedule.anneal_times == 1:
        return [schedule.start_strength]
    return [
        schedule.start_strength
        + (schedule.end_strength - schedule.start_strength)
        * i
        / (schedule.anneal_times - 1)
        for i in range(schedule.anneal_times)
    ]


def run_one_edit_pass(
    state: TrainingState,
    pipe,
    step_idx: int,
    edit_idx: int,
    strength: float,
    guidance_scale: float,
    add_prompt: str = "",
):
    scene = state.scene
    gaussians = state.gaussians
    sorted_idx = state.sorted_idx
    background = state.background

    with torch.no_grad():
        rendered_images = render_sets_train(
            scene.getTrainCameras().copy(), gaussians, pipe, background
        )
        rendered_images = torch.stack(rendered_images, dim=0)

        images, weight = state.model.img2img(
            rendered_images,
            imgs2=state.original_images,
            strength=strength,
            add_prompt=add_prompt,
            guidance_scale=guidance_scale,
        )

        for idx, view in enumerate(scene.getTrainCameras()):
            view.original_image = images[idx]
            view.weight = weight[idx]


def optimize_gaussians_for_edits(
    state: TrainingState,
    pipe,
    opt,
    args,
    schedule: DragSchedule,
    saving_iterations,
    checkpoint_iterations,
    total_iterations: int,
    progress_bar: tqdm,
    global_iter: int,
):
    scene = state.scene
    gaussians = state.gaussians
    p_ssim = state.p_ssim
    original_images = state.original_images
    background = state.background
    original_cameras = state.original_cameras

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    viewpoint_stack = None

    for _ in range(schedule.train_iterations_per_edit):
        global_iter += 1

        iter_start.record()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(
            randint(0, len(viewpoint_stack) - 1)
        )

        if (global_iter - 1) == args.debug_from:
            pipe.debug = True

        bg = (
            torch.rand(3, device="cuda")
            if opt.random_background
            else background
        )

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        view_mask = viewpoint_cam.viewmask.cuda().float()

        loss = 0
        loss += p_ssim(image, gt_image)
        loss += l1_loss(image, gt_image)


        bg_gt_image = original_images[viewpoint_cam.idx].cuda()
        ssim_loss = 1 - ssim(image, bg_gt_image, mask=view_mask)

        inv_mask = 1.0 - view_mask
        masked_image = image * inv_mask
        masked_gt_image = bg_gt_image * inv_mask
        Ll1 = l1_loss(masked_image, masked_gt_image)

        bg_loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * ssim_loss
        )
        bg_loss *= 100.0
        loss += bg_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
            progress_bar.update(1)

            if global_iter in saving_iterations:
                print(f"\n[ITER {global_iter}] Saving Gaussians")
                scene.save(global_iter)


            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if global_iter in checkpoint_iterations:
                print(f"\n[ITER {global_iter}] Saving Checkpoint")
                ckpt_path = os.path.join(
                    scene.model_path, f"chkpnt{global_iter}.pth"
                )
                print(f"save_path:{ckpt_path}")
                torch.save((gaussians.capture(), global_iter), ckpt_path)

            if global_iter % 500 == 0 or global_iter == total_iterations:
                with torch.no_grad():
                    real_images = render_sets_train(
                        original_cameras, gaussians, pipe, background
                    )
                    save_dir = os.path.join(
                        scene.model_path, "render_real", f"iter_{global_iter}"
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    for idx, img in enumerate(real_images):
                        torchvision.utils.save_image(
                            img, f"{save_dir}/{idx}.png"
                        )
                    scene.save(global_iter)

    return global_iter


def run_drag_training_loop(
    state: TrainingState,
    schedule: DragSchedule,
    opt,
    args,
    saving_iterations,
    checkpoint_iterations,
    pipe,
):
    total_iterations = (
        schedule.drag_steps
        * schedule.anneal_times
        * schedule.train_iterations_per_edit
    )
    opt.iterations = total_iterations

    progress_bar = tqdm(total=total_iterations, desc="Training progress")
    global_iter = 0

    for step_idx in range(schedule.drag_steps):
        apply_one_drag_step(state, pipe, step_idx)
        strengths = compute_step_strengths(schedule)

        for edit_idx, st in enumerate(strengths):
            run_one_edit_pass(
                state,
                pipe,
                step_idx=step_idx,
                edit_idx=edit_idx,
                strength=st,
                guidance_scale=schedule.guidance_scale,
                add_prompt="",
            )

            global_iter = optimize_gaussians_for_edits(
                state,
                pipe,
                opt,
                args,
                schedule,
                saving_iterations,
                checkpoint_iterations,
                total_iterations,
                progress_bar,
                global_iter,
            )

    progress_bar.close()


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    state, schedule = init_scene_and_gaussians(dataset, opt, args, pipe)
    run_drag_training_loop(
        state,
        schedule,
        opt,
        args,
        saving_iterations,
        checkpoint_iterations,
        pipe,
    )


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6010)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000]
    )

    parser.add_argument(
        "--start_checkpoint", type=str, default=None, help="Path to a training checkpoint (.pth)"
    )
    parser.add_argument(
        "--ply", type=str, default=None, help="Path to an initial Gaussian .ply file"
    )
    parser.add_argument(
        "--drag_path",
        type=str,
        required=True,
        help="Path to the drag config JSON file",
    )

    parser.add_argument(
        "--drag_steps",
        type=int,
        default=2,
        help="Number of drag+edit steps (discretization of motion).",
    )
    parser.add_argument(
        "--anneal_times",
        type=int,
        default=4,
        help="Number of editing passes per step (strength annealing).",
    )
    parser.add_argument(
        "--start_strength",
        type=float,
        default=0.5,
        help="Starting img2img strength per step.",
    )
    parser.add_argument(
        "--end_strength",
        type=float,
        default=0.3,
        help="Ending img2img strength per step.",
    )
    parser.add_argument(
        "--train_iterations_per_edit",
        type=int,
        default=500,
        help="Gaussian training iterations after each edit.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="DreamBooth img2img guidance scale.",
    )
    parser.add_argument(
        "--camera_num",
        type=int,
        default=50,
        help="Number of training cameras to select.",
    )

    return parser, lp, op, pp


def main():
    parser, lp, op, pp = build_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing", getattr(args, "model_path", ""))

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    training(
        dataset,
        opt,
        pipe,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
