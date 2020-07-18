#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Descripttion: https://github.com/sxhxliang/detectron2_backbone
# version: 0.0.1
# Author: Shihua Liang (sxhx.liang@gmail.com)
# FilePath: /detectron2_backbone/detectron2_backbone/tools/convert_weight.py
# Create: 2020-05-05 07:32:08
# LastAuthor: Shihua Liang
# lastTime: 2020-07-02 21:51:57
# --------------------------------------------------------
import torch
import argparse
from collections import OrderedDict

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Model Converter")
    parser.add_argument(
        "--model",
        required=True,
        metavar="FILE",
        help="path to model weights",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="path to model weights",
    )

    parser.add_argument(
        "--mbv2",
        required=False,
        action='store_true',
        help="remap the mbv2 to flatten each layer",
    )
    return parser

features_1_mapping = {
    'conv.0.0': 'conv.0',
    'conv.0.1': 'conv.1',
    'conv.1': 'conv.3',
    'conv.2': 'conv.4',
}
features_2_17_mapping = {
    'conv.0.0': 'conv.0',
    'conv.0.1': 'conv.1',
    'conv.1.0': 'conv.3',
    'conv.1.1': 'conv.4',
    'conv.2': 'conv.6',
    'conv.3': 'conv.7',
}
mbv2_mapping = {
    'features.1.': features_1_mapping,
    'features.2.': features_2_17_mapping,
    'features.3.': features_2_17_mapping,
    'features.4.': features_2_17_mapping,
    'features.5.': features_2_17_mapping,
    'features.6.': features_2_17_mapping,
    'features.7.': features_2_17_mapping,
    'features.8.': features_2_17_mapping,
    'features.9.': features_2_17_mapping,
    'features.10.': features_2_17_mapping,
    'features.11.': features_2_17_mapping,
    'features.12.': features_2_17_mapping,
    'features.13.': features_2_17_mapping,
    'features.14.': features_2_17_mapping,
    'features.15.': features_2_17_mapping,
    'features.16.': features_2_17_mapping,
    'features.17.': features_2_17_mapping,
}

def rename_for_mbv2(old):
    import collections
    renamed = collections.OrderedDict()
    for k, v in old.items():
        new_name = k
        for prefix, nm in mbv2_mapping.items():
            if k.startswith(prefix):
                found = False
                for middle, new_middle in nm.items():
                    if k.startswith(prefix+middle):
                        found = True
                        new_name = k.replace(middle, new_middle)
                        break
                assert found, k
                break
        renamed[new_name]= v
    return renamed
def convert_weight():
    args = get_parser().parse_args()
    ckpt = torch.load(args.model, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    if args.mbv2:
        state_dict = rename_for_mbv2(state_dict)
    model = {"model": state_dict, "__author__": "custom", "matching_heuristics": True}

    torch.save(model, args.output)

if __name__ == "__main__":
    convert_weight()
