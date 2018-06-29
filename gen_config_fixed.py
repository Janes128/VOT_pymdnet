import os
import numpy as np


def gen_config_fixed(path):
    if path != '':
        # generate config from a sequence name

        seq_home = '../dataset/OTB'
        save_home = '../result_fig'
        result_home = '../result'

        seq_name = path
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        gt = np.loadtxt(gt_path, delimiter=',')
        init_bbox = gt[0]

    return init_bbox