import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from models.adaptive_controller import AdaptiveController
from IPython.display import display, SVG


def load_renderer(args, target_im=None, mask=None, num_strokes=None):
    strokes = num_strokes if num_strokes is not None else args.num_paths
    renderer = Painter(num_strokes=strokes, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        new_image = Image.new("RGBA", target.size, "WHITE")
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    
    # 如果提供了外部 mask，使用外部 mask
    if args.external_mask and os.path.exists(args.external_mask):
        print(f"Using external mask: {args.external_mask}")
        mask = Image.open(args.external_mask)
        mask = mask.convert('L')
        mask = np.array(mask) / 255.0
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        masked_im = target  # 使用输入图像作为目标
    else:
        # 默认行为：使用 U2Net 提取 mask
        masked_im, mask = utils.get_mask_u2net(args, target)
    
    if args.mask_object:
        target = masked_im
    
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        scale_factor = args.image_scale / max(target.size[0], target.size[1])
        new_width = int(target.size[0] * scale_factor)
        new_height = int(target.size[1] * scale_factor)
        transforms_.append(transforms.Resize(
            (new_height, new_width), interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
    
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    
    return target_, mask


def train_stage(args, renderer, loss_func, optimizer, start_epoch, num_iter, adaptive_controller=None, mask=None, inputs=None, loss_eval_list=None):
    counter = start_epoch
    best_loss = 100
    best_iter = 0
    min_delta = 1e-5
    terminate = False

    if args.display:
        epoch_range = range(num_iter)
    else:
        epoch_range = tqdm(range(num_iter))

    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        start = time.time()
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)
        losses_dict = loss_func(sketches, inputs.detach(
        ), renderer.get_color_parameters(), renderer, counter, optimizer, mask=mask)
        loss = sum(list(losses_dict.values()))
        if getattr(args, "mask_penalty", 0.0) > 0 and mask is not None:
            loss = loss + args.mask_penalty * renderer.mask_penalty()
        loss.backward()
        optimizer.step_()

        if epoch % args.save_interval == 0:
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{counter}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{counter}")

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                ), renderer.get_points_parans(), counter, optimizer, mode="eval", mask=mask)
                loss_eval = sum(list(losses_dict_eval.values()))
                if loss_eval_list is not None:
                    loss_eval_list.append(loss_eval.item())

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = counter
                        terminate = False
                        utils.plot_batch(
                            inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_eval.keys():
                        wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                    wandb.log(wandb_dict, step=counter)

                if adaptive_controller:
                    should_stop = adaptive_controller.should_stop(
                        sketches, renderer.strokes_counter)
                    controller_state = adaptive_controller.get_state()
                    
                    if args.use_wandb:
                        wandb.log({
                            "semantic_score": controller_state['best_semantic_score'],
                            "complexity_score": controller_state['best_complexity_score'],
                            "convergence_rate": controller_state['convergence_rate'],
                            "current_strokes": controller_state['current_strokes'],
                            "adaptive_factor": controller_state['adaptive_factor']
                        }, step=counter)

                    if should_stop:
                        return True, counter, best_loss, best_iter

                if abs(cur_delta) <= min_delta:
                    if terminate:
                        break
                    terminate = True

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)

        counter += 1

    return False, counter, best_loss, best_iter


def main(args):
    global inputs
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir, getattr(args, 'rgba_target', None))

    configs_to_save = {"loss_eval": [], "stage_info": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0

    adaptive_controller = None
    if args.adaptive_stopping:
        adaptive_controller = AdaptiveController(args, inputs)
        current_strokes = args.min_strokes
    else:
        current_strokes = args.num_paths

    renderer = load_renderer(args, inputs, mask, num_strokes=current_strokes)
    optimizer = PainterOptimizer(args, renderer)
    renderer.set_random_noise(0)
    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()

    total_counter = 0
    stage = 0
    max_stages = (args.max_strokes // args.stroke_increment) + 1

    while stage < max_stages:
        print(f"\n=== Stage {stage + 1}: Training with {current_strokes} strokes ===")
        
        should_terminate, final_counter, stage_best_loss, stage_best_iter = train_stage(
            args, renderer, loss_func, optimizer, total_counter, args.num_iter, adaptive_controller, mask=mask, inputs=inputs,
            loss_eval_list=configs_to_save["loss_eval"]
        )
        
        total_counter = final_counter
        
        if should_terminate:
            print(f"Adaptive stopping triggered at stage {stage + 1}")
            break

        if not args.adaptive_stopping:
            if stage >= args.num_stages - 1:
                break

        if adaptive_controller:
            if adaptive_controller.should_add_strokes():
                increment = adaptive_controller.get_stroke_increment()
                new_strokes = current_strokes + increment
                
                if new_strokes > args.max_strokes:
                    new_strokes = args.max_strokes
                
                print(f"Adding {increment} strokes (total: {new_strokes})")
                
                renderer.num_paths = new_strokes
                renderer.strokes_per_stage = increment
                
                img = renderer.init_image(stage=stage + 1)
                
                optimizer = PainterOptimizer(args, renderer)
                optimizer.init_optimizers()
                
                current_strokes = new_strokes
                stage += 1
            else:
                print("No need to add more strokes. Stopping early.")
                break
        else:
            stage += 1

    renderer.save_svg(args.output_dir, "final_svg")
    
    if adaptive_controller and adaptive_controller.best_sketch is not None:
        renderer.save_svg(args.output_dir, "best_adaptive")
        utils.plot_batch(inputs, adaptive_controller.best_sketch, args.output_dir, 
                         total_counter, use_wandb=args.use_wandb, title="best_adaptive.jpg")
        
        if args.use_wandb:
            wandb.run.summary["best_semantic_score"] = adaptive_controller.best_semantic_score
            wandb.run.summary["final_strokes"] = current_strokes

    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(
        path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

    return configs_to_save


if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()