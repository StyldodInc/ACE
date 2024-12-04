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
        sampler=pipe.input.get("sampler", "ddim"),
        sample_steps=pipe.input.get("sample_steps", 20),
        guide_scale=pipe.input.get("guide_scale", 4.5),
        guide_rescale=pipe.input.get("guide_rescale", 0.5),
        seed=seed,
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
                        default=None)
    parser.add_argument('--output_w',
                        dest='output_w',
                        help='The width of output image for generation tasks!',
                        type=int,
                        default=None)
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
    pipe = ACEInference()
    pipe.init_from_cfg(cfg)


    output_h = cfg.args.output_h or pipe.input.get("output_height", 1024)
    output_w = cfg.args.output_w or pipe.input.get("output_width", 1024)
    negative_prompt = cfg.args.negative_prompt

    if cfg.args.instruction == "" and cfg.args.input_image is None:
        # run examples
        all_examples = [
            ["examples/input_images/example0.webp", None, "",
             "{image} make the boy cry, his eyes filled with tears",
             "", 199999, output_h, output_w, "examples/output_images/example0.png"],
            ["examples/input_images/example1.webp", None, "",
             "{image}use the depth map @cb638863a0e9 and the text caption  \"Vincent van Gogh with expressive, "
             "soulful eyes and a gentle smile, wearing traditional 19th-century artist's attire, including a "
             "paint-streaked smock, a straw hat with sunflowers, and an artist's easel slung over his shoulder."
             "Subtle elements of \"Starry Night\" swirling around, with hints of sunflowers and wheat fields "
             "from his famous paintings. Include a palette and paintbrushes, a small sun painted in the top "
             "corner, and subtle curling patterns reminiscent of his brush strokes\" to create a image",
             "", 899999, output_h, output_w, "examples/output_images/example1.png"],
            ["examples/input_images/example2.webp", None, "",
             "make this {image} colorful",
             "", 199999, output_h, output_w, "examples/output_images/example2.png"],
            ["examples/input_images/example3.webp", None, "",
             "change the style to 3D cartoon style",
             "", 2023, output_h, output_w, "examples/output_images/example3.png"],

        ]
        for example in all_examples:
            run_one_case(pipe, example[0], example[1], example[2], example[3],
                         example[4], example[5], example[6], example[7], example[8])
    else:
        if "{image}" not in cfg.args.instruction:
            instruction = "{image} " + cfg.args.instruction
        else:
            instruction = cfg.args.instruction

        run_one_case(pipe, cfg.args.input_image, cfg.args.input_mask, "",
                 instruction, negative_prompt, cfg.args.seed,
                 output_h, output_w, cfg.args.save_path)

if __name__ == '__main__':
    run()
