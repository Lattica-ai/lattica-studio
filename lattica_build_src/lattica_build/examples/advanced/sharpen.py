from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from lattica_build.operators.client_ops import Clamp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.ml.h_conv import HomConv
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.params.params import HomParams, DecompositionType

# repeat kernel across groups
SHARPEN_KERNEL = torch.tensor([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]).repeat(3, 1, 1, 1).to(torch.float32)


def build_pipeline(img_size=100) -> HomomorphicPipeline:
    input_shape = (3, img_size, img_size)

    hom_pipeline = HomomorphicPipeline(
        client_pre=[
            HomReshape((3, img_size * img_size))
        ],
        hom=HomConv(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3),
        client_post=[
            HomReshape(input_shape),
            Clamp(0, 1)
        ],
        n_axis=-1,
        input_shape=input_shape,
    )
    hom_pipeline.set_data(0, SHARPEN_KERNEL)

    return hom_pipeline

def build_params():
    return HomParams(
        n=2 ** 15,
        full_q_list_precision=(
            (60, 30,),
        ),
        pt_scale=2 ** 30,
        num_special_primes=1
    )
