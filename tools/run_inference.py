########      Modified by rohit.panda on 22 Nov 2024      ############

# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import importlib
import io
import os
import sys
from PIL import Image
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

from chatbot.ace_inference import ACEInference

fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]

for one_fs in fs_list:
    FS.init_fs_client(one_fs)


def run_one_case(pipe, input_image, input_mask, edit_k,
                 instruction, negative_prompt, seed,
                 output_h, output_w, save_path):
    edit_image, edit_image_mask, edit_task = [], [], []
    if input_image is not None:
        image = Image.open(io.BytesIO(FS.get_object(input_image)))
        edit_image.append(image.convert('RGB'))
        edit_image_mask.append(
            Image.open(Image.open(io.BytesIO(FS.get_object(input_mask)))).
            convert('L') if input_mask is not None else None)
        edit_task.append(edit_k)
    imgs = pipe(
        image=edit_image,
        mask=edit_image_mask,
        task=edit_task,
        prompt=[instruction] *
               len(edit_image) if edit_image is not None else [instruction],
        negative_prompt=[negative_prompt] * len(edit_image)
        if edit_image is not None else [negative_prompt],
        output_height=output_h,
        output_width=output_w,
        seed=seed
    )
    with FS.put_to(save_path) as local_path:
        imgs[0].save(local_path)
    return


def run():
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--instruction',
                        dest='instruction',
                        help='The instruction for editing or generating!',
                        default="")
    parser.add_argument('--negative_prompt',
                        dest='negative_prompt',
                        help='The negative prompt for editing or generating!',
                        default="")
    parser.add_argument('--output_h',
                        dest='output_h',
                        help='The height of output image for generation tasks!',
                        type=int,
                        default=512)
    parser.add_argument('--output_w',
                        dest='output_w',
                        help='The width of output image for generation tasks!',
                        type=int,
                        default=512)
    parser.add_argument('--input_image',
                        dest='input_image',
                        help='The input image!',
                        default=None
                        )
    parser.add_argument('--input_mask',
                        dest='input_mask',
                        help='The input mask!',
                        default=None
                        )
    parser.add_argument('--input_path'
                        )
    # parser.add_argument('--cfg'
    #                     )
    parser.add_argument('--save_path',
                        dest='save_path',
                        help='The save path for output image!',
                        default='examples/output_images/output.png'
                        )
    parser.add_argument('--seed',
                        dest='seed',
                        help='The seed for generation!',
                        type=int,
                        default=-1)
    cfg = Config(load=True, parser_ins=parser)
    # cfg = Config(load=True, cfg_file=cfg.args)
    pipe = ACEInference()
    pipe.init_from_cfg(cfg)


    output_h = cfg.args.output_h or pipe.input.get("output_height", 1024)
    output_w = cfg.args.output_w or pipe.input.get("output_width", 1024)
    negative_prompt = cfg.args.negative_prompt

    # if "{image}" not in cfg.args.instruction:
    #     instruction = "{image} " + cfg.args.instruction
    # else:
    instruction = cfg.args.instruction

    os.makedirs(cfg.args.save_path, exist_ok=True)
    all_examples = os.listdir(cfg.args.input_path)
    for example in all_examples:

        run_one_case(pipe, os.path.join(cfg.args.input_path, example), cfg.args.input_mask, "",
                instruction, negative_prompt, cfg.args.seed,
                output_h, output_w, os.path.join(cfg.args.save_path, example))

if __name__ == '__main__':
    run()

