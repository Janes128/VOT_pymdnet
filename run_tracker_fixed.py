import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

# Jane add (2018/06/25)
import cv2
global pic_num

# Parameters
drawing = False  # true if mouse pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1  # init target position
isDraw = False

sys.path.insert(0, '../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *
from showResult import *

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        # print "Iter %d, Loss %.4f" % (iter, loss.data[0])

# Jane add (2018/6/25)
# mouse callback function
def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, mode, isDraw
    # is Click
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        print("ix= ", ix, "iy= ", iy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(first_image, (ix, iy), (x, y), (0, 255, 255), 2)
            tempx = min(ix, x)
            tempy = min(iy, y)
            tempw = abs(ix - x)
            temph = abs(iy - y)

            # Input the groundtruth
            initfile = open('C:/Users/User/py-MDNet/dataset/OTB/Save/groundtruth_rect.txt', 'w')
            initfile_str = str(tempx) + ',' + str(tempy) + ',' + str(tempw) + ',' + str(temph)
            # for j in range(pic_num - 1):
            #     initfile_str += '\n0,0,0,0'
            initfile.write(initfile_str)
            isDraw = True
            # time.sleep(1)
            cv2.destroyAllWindows()
            # ss.screenShot(int(pic_num))

def gen_config_fixed(path):
    if path != '':
        # generate config from a sequence name

        seq_home = '../dataset/OTB'

        seq_name = path
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        gt = np.loadtxt(gt_path, delimiter=',')
        init_bbox = gt

    return init_bbox

def run_mdnet(img_list, init_bbox):
    # img_list, init_bbox, gt=None, display=False
    display = True
    isSave = True
    save_dir = 'C:/Users/User/py-MDNet/dataset/OTB/Save/result'

    print("Hello")
    print("init bbox", init_bbox)
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image = Image.fromarray(img_list[0]).convert('RGB')
    # image = img_list[0]
    print("image size: ", image.size)
    # Train bbox regressor
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)
    feat_dim = pos_feats.size(-1)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])

    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]

    spf_total = time.time() - tic

    # Display
    savefig = save_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)  # im = ax.imshow(image, aspect='normal')

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
            linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
            plt.plot()
            plt.show()
        if savefig and isSave:
            fig.savefig(os.path.join(save_dir,'0000.jpg'),dpi=dpi)
        plt.show(block=False)

    # Main loop
    for i in range(1, len(img_list)):

        tic = time.time()
        # cv2.imshow("sdfasdf", img_list[i])
        # print("The #", i, "Picture", img_list[i])
        print("The #", i, "Picture")
        # Load image
        image = Image.fromarray(img_list[i]).convert('RGB')
        # image = img_list[i].convert('RGB')

        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Copy previous result at failure
        if not success:
            target_bbox = result[i - 1]
            bbreg_bbox = result_bb[i - 1]

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
                plt.show()
            if savefig and isSave:
                fig.savefig(os.path.join(save_dir, '%04d.jpg' % (i)), dpi=dpi)
            # plt.show(block=False)

        print("Frame %d/%d, Score %.3f, Time %.3f" % \
            (i, len(img_list), target_score, spf))
        # else:
        #     print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
        #           (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))

    fps = len(img_list) / spf_total
    return result, result_bb, fps

if __name__ == "__main__":
    # Generate sequence config
    # img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # parameter
    display = True

    print("Welcome! Using pyMDNet VOT project...")
    print("Draw the first bbox: ")
    time.sleep(1)

    video_home = 'D:\\pyMDNET\\video/pigTrim_2.mp4'  # the path of video
    video = cv2.VideoCapture(video_home)

    retval, first_image = video.read()
    cv2.namedWindow('Init_image')
    cv2.imshow("Init_image", first_image)
    # print("The first Image type: ", first_image.type)
    # im = Image.fromarray(first_image, 'RGB')                # Convert the narray to image
    # im = np.array(im)
    # cv2.imshow("Testing", im)
    # time.sleep(10)

    cv2.setMouseCallback('Init_image', draw_rect)

    # and (past_local == current_local)
    while (1 and isDraw):
        cv2.imshow('Init_image', first_image)
        k = cv2.waitKey(1) & 0xFF  # Waiting key and make sure that it's at least 8 bits
        if k == ord('m'):
            mode = not mode
        elif k == 27:  # Esc key to stop
            break

    print("Save init_bbox...")
    init_bbox = gen_config_fixed('Save')
    print("I'm bbox: ", init_bbox)

    img_list = []

    print("Save the picture...")
    img_list.append(first_image)

    # Save the other picture
    while cv2.waitKey(30) != ord('q'):
        retval, image_other = video.read()
        if not retval:
            break
        img_list.append(image_other)
        # cv2.imshow("video", image_other)
    print("Finish Save.")
    video.release()

    # print("Get the test picture: ")
    #image_test = Image.open("D:/pyMDNET/Temp/img/0001.jpg").convert('RGB')

    print("run the mdnet...")
    # Run tracker
    result, result_bb, fps = run_mdnet(img_list, init_bbox)

    print("Finish saving result...")
    isShow = False
    input("Do you want to show the result", isShow)

    if isShow:
        showResult('C:/Users/User/py-MDNet/dataset/OTB/Save/result')

