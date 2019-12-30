"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import torch
print(torch.version.__version__)
from process_stylization import stylization
from photo_wct import PhotoWCT
from photo_smooth import Propagator
import random
import os
import json

print(torch.cuda.get_device_name(0))

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model_path', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--content_image_path', default='sleeba.jpg')
parser.add_argument('--style_image_path', default='./style_images')
parser.add_argument('--output_image_path', default='./output/example1.png')
args = parser.parse_args()

if not (args.output_image_path is None):
    os.makedirs(args.output_image_path, exist_ok=True)
print("Output folder created!")

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model_path))
p_pro = Propagator()
p_wct.cuda(0)
style_images_list = os.listdir(args.style_image_path)
style_images_map = {}
content_image = os.path.basename(args.content_image_path)
    
style_image = random.choice(style_images_list)
style_images_map[content_image] = style_image
stylization(
    stylization_module=p_wct,
    smoothing_module=p_pro,
    content_image_path=args.content_image_path + "/" + "0_s_0_g2.png",
    style_image_path=args.style_image_path + "/" + style_image,
    content_seg_path=[],
    style_seg_path=[],
    output_image_path=args.output_image_path,
    cuda=1,
    save_intermediate=True,
    no_post=False)
print("\n{} is done\n".format(args.content_image_path))
print("+++"*20)

with open(args.output_image_path +  '/style_images_map.json', 'w') as fp:
    json.dump(style_images_map, fp)
