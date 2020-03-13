import torch
import numpy as np
from scipy.io import savemat

import time
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import pdb


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors) // num_anchors
    conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask = torch.zeros(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW)

    nAnchors = nA * nH * nW
    nPixels = nH * nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            cur_gt_boxes = torch.FloatTensor(
                [gx, gy, gw, gh]).repeat(nAnchors, 1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(
                cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        # conf_mask[b][cur_ious>sil_thresh] = 0
        conf_mask[b][torch.reshape(cur_ious, (nA, nH, nW)) > sil_thresh] = 0
    if seen < 12800:
        if anchor_step == 4:
            tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(
                1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
            ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(
                1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
        else:
            tx.fill_(0.5)
            ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = [0, 0, aw, ah]
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step * n + 2]
                    ay = anchors[anchor_step * n + 3]
                    dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b * nAnchors +
                                  best_n * nPixels + gj * nW + gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t * 5 + 1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t * 5 + 2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(
                gw / anchors[anchor_step * best_n])
            th[b][best_n][gj][gi] = math.log(
                gh / anchors[anchor_step * best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t * 5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        self.loss = AverageMeter()
        self.loss_x = AverageMeter()
        self.loss_y = AverageMeter()
        self.loss_w = AverageMeter()
        self.loss_h = AverageMeter()
        self.loss_conf = AverageMeter()
        self.loss_cls = AverageMeter()
        self.loss_def = AverageMeter()
        self.lambda1 = 5.0

    def forward(self, output, target, features):

        output_orig = output.clone()

        # output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB, nA, (5 + nC), nH, nW)
        x = F.sigmoid(output.index_select(2, Variable(
            torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y = F.sigmoid(output.index_select(2, Variable(
            torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w = output.index_select(2, Variable(
            torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h = output.index_select(2, Variable(
            torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(
            torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls = output.index_select(2, Variable(
            torch.linspace(5, 5 + nC - 1, nC).long().cuda()))
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1,
                                                       2).contiguous().view(nB * nA * nH * nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
        grid_x = torch.linspace(
            0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(
            0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(
            nA, int(self.anchor_step)).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(
            nA, int(self.anchor_step)).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(
            1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(
            1, 1, nH * nW).view(nB * nA * nH * nW)
        pred_boxes[0] = torch.reshape(x.data, (1, nB * nA * nH * nW)) + grid_x
        pred_boxes[1] = torch.reshape(y.data, (1, nB * nA * nH * nW)) + grid_y
        pred_boxes[2] = torch.reshape(
            torch.exp(w.data), (1, nB * nA * nH * nW)) * anchor_w
        pred_boxes[3] = torch.reshape(
            torch.exp(h.data), (1, nB * nA * nH * nW)) * anchor_h

        pred_boxes_orig = pred_boxes.transpose(
            0, 1).contiguous().view(-1, 4).clone()

        pred_boxes = convert2cpu(
            pred_boxes.transpose(0, 1).contiguous().view(-1, 4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC,
                                                                                                    nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().item())

        cls_orig = cls.clone()
        cls_mask_orig = cls_mask.clone()

        tx = Variable(tx.cuda())
        ty = Variable(ty.cuda())
        tw = Variable(tw.cuda())
        th = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls = tcls[cls_mask == 1].view(-1).long().cuda()

        coord_mask = Variable(coord_mask.cuda())
        conf_mask = Variable(conf_mask.cuda().sqrt())
        cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())

        cls = cls[cls_mask == 1].view(-1, nC)

        t3 = time.time()

        loss_x = self.coord_scale * \
            nn.MSELoss(size_average=False)(
                x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * \
            nn.MSELoss(size_average=False)(
                y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * \
            nn.MSELoss(size_average=False)(
                w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * \
            nn.MSELoss(size_average=False)(
                h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(size_average=False)(
            conf * conf_mask, tconf * conf_mask) / 2.0
        if cls.size(0) == 0 and tcls.size(0) == 0:
            loss_cls = torch.Tensor([0])
        else:
            loss_cls = self.class_scale * \
                nn.CrossEntropyLoss(size_average=False)(cls, tcls)


# defense loss begin

        # indices of detections matched to gt
        inds = torch.nonzero(cls_mask_orig)
        sz_hwa = nH * nW * nA
        sz_hw = nH * nW
        nHf = features.data.size(2)
        nWf = features.data.size(3)

        cls_max_confs, cls_max_ids = torch.max(F.softmax(cls_orig), 1)

        pred_boxes2 = pred_boxes_orig.clone()
        # resize to activation size.
        pred_boxes2[:, 0] = pred_boxes_orig[:, 0] / nW * nWf
        pred_boxes2[:, 1] = pred_boxes_orig[:, 1] / nH * nHf
        pred_boxes2[:, 2] = pred_boxes_orig[:, 2] / nW * nWf
        pred_boxes2[:, 3] = pred_boxes_orig[:, 3] / nH * nHf

        # get the corners of boxes rather than center
        pred_boxes2_corners = torch.zeros_like(pred_boxes_orig).long()
        pred_boxes2_corners[:, 0] = torch.floor(torch.max(
            pred_boxes2[:, 0] - pred_boxes2[:, 2] / 2, torch.zeros_like(pred_boxes2[:, 0] - pred_boxes2[:, 2] / 2))).long()
        pred_boxes2_corners[:, 1] = torch.floor(torch.max(
            pred_boxes2[:, 1] - pred_boxes2[:, 3] / 2, torch.zeros_like(pred_boxes2[:, 1] - pred_boxes2[:, 3] / 2))).long()
        pred_boxes2_corners[:, 2] = torch.ceil(torch.min(
            pred_boxes2[:, 0] + pred_boxes2[:, 2] / 2 + 1e-5, nHf * torch.ones_like(pred_boxes2[:, 0] + pred_boxes2[:, 2] / 2))).long()
        pred_boxes2_corners[:, 3] = torch.ceil(torch.min(
            pred_boxes2[:, 1] + pred_boxes2[:, 3] / 2 + 1e-5, nWf * torch.ones_like(pred_boxes2[:, 1] + pred_boxes2[:, 3] / 2))).long()

        loss_def = torch.Tensor([0]).cuda()
        for i in range(inds.size(0)):
            ind = inds[i, 0] * sz_hwa + inds[i, 1] * \
                sz_hw + inds[i, 2] * nW + inds[i, 3]
            this_im_ind = inds[i, 0]
            score = cls_max_confs[ind] * conf[inds[i, 0],
                                              inds[i, 1], inds[i, 2], inds[i, 3]]

            dy_dz, = torch.autograd.grad(score, features, grad_outputs=torch.ones(
                score.size()).cuda(), retain_graph=True, create_graph=True)

            dydA_sumk1 = (dy_dz.abs().sum(dim=1)) + 1e-5
            dydA_sumk2 = dydA_sumk1 / (dydA_sumk1.sum(dim=1).sum(dim=1).unsqueeze(1).repeat(
                1, nHf).unsqueeze(2).repeat(1, 1, nWf))  # normalized  nB x nHf x nWf
            dydA_sumk3 = dydA_sumk2[this_im_ind, pred_boxes2_corners[ind, 1]:pred_boxes2_corners[ind, 3], pred_boxes2_corners[ind, 0]:pred_boxes2_corners[ind, 2]]
            loss_tmp = dydA_sumk2[this_im_ind, :, :].sum(
                dim=0).sum(dim=0) - dydA_sumk3.sum(dim=0).sum(dim=0)
            loss_def += loss_tmp

# defense loss end
        loss_def = self.lambda1 * loss_def
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls + loss_def

        self.loss.update(loss.clone().detach(
        ).item(), output.clone().detach().data.size(0))
        self.loss_x.update(loss_x.clone().detach(
        ).item(), output.clone().detach().data.size(0))
        self.loss_y.update(loss_y.clone().detach(
        ).item(), output.clone().detach().data.size(0))
        self.loss_w.update(loss_w.clone().detach(
        ).item(), output.clone().detach().data.size(0))
        self.loss_h.update(loss_h.clone().detach(
        ).item(), output.clone().detach().data.size(0))
        self.loss_conf.update(loss_conf.clone().detach(
        ).item(), output.clone().detach().data.size(0))
        self.loss_cls.update(loss_cls.clone().detach(
        ).item(), output.clone().detach().data.size(0))
        self.loss_def.update(loss_def.clone().detach(
        ).item(), output.clone().detach().data.size(0))

        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        print('{0}: nGT {1}, proposals {2}, loss: x {loss_x.val:.3f}({loss_x.avg:.3f}), y {loss_y.val:.3f}({loss_y.avg:.3f}), w {loss_w.val:.3f}({loss_w.avg:.3f}), h {loss_h.val:.3f}({loss_h.avg:.3f}), conf {loss_conf.val:.3f}({loss_conf.avg:.3f}), cls {loss_cls.val:.3f}({loss_cls.avg:.3f}), def {loss_def.val:.3f}({loss_def.avg:.3f}), total {loss.val:.3f}({loss.avg:.3f})'.format(
            self.seen, nGT, nProposals, loss_x=self.loss_x, loss_y=self.loss_y, loss_w=self.loss_w, loss_h=self.loss_h, loss_conf=self.loss_conf, loss_cls=self.loss_cls, loss_def=self.loss_def, loss=self.loss))

        return loss
