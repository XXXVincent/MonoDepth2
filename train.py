# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions, NuscdepthOptions
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
options = NuscdepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
