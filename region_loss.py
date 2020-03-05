import time
import torch
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
    anchor_step = len(anchors)/num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW)
    ty         = torch.zeros(nB, nA, nH, nW)
    tw         = torch.zeros(nB, nA, nH, nW)
    th         = torch.zeros(nB, nA, nH, nW)
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW)

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in xrange(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in xrange(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask[b][cur_ious>sil_thresh] = 0
    if seen < 12800:
       if anchor_step == 4:
           tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5)
           ty.fill_(0.5)
       tw.zero_()
       th.zero_()
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        for t in xrange(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in xrange(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0


    def forward(self, output, target, reqd_class_index):
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors,\
                                                                         nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)



        cls_mask = (cls_mask == 1)

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())

        '''
            Consider all detections returned by cls_mask. Take the probabilities for the target class for
            all those detections.
        '''
        cls      = cls[cls_mask].view(-1, nC)
        cls_prob = F.softmax(cls)                  # - Change Aniruddha
        reqd_class_index_var = Variable(torch.LongTensor([reqd_class_index]).cuda())
        cls_prob2 = torch.gather(cls_prob, 1, reqd_class_index_var.repeat(cls_prob.size(0),1))
        target0 = torch.zeros(cls_prob2.size()[0], 1)
        target0 = Variable(target0.cuda())
        loss = nn.L1Loss(size_average=True)(cls_prob2, target0)
        max_class_prob, _ = torch.max(cls_prob2,0)

        t3 = time.time()

        return loss, max_class_prob.data[0]

class RegionLoss_BlindnessAttack(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss_BlindnessAttack, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0


    def forward(self, output, target, blindness_class_index):

        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors,\
                                                                         nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)



        cls_mask = (cls_mask == 1)

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())

        cls      = cls[cls_mask].view(-1, nC)
        cls_prob = F.softmax(cls)
        blindness_class_index_var = Variable(torch.LongTensor([blindness_class_index]).cuda())

        # gather logits and probabilities of blindness class
        blindness_class_logits = torch.gather(cls, 1, blindness_class_index_var.repeat(cls.size(0), 1))
        blindness_class_probs = torch.gather(cls_prob, 1, blindness_class_index_var.repeat(cls.size(0), 1))

        '''
        dummy, ind = cls.max(1)
        tmp1 = blindness_class_logits[0] - cls.sum(dim=1)
        tmp2 = (ind==blindness_class_index)
        tmp3 = tmp1*tmp2.float()
        loss = tmp3.sum()
        '''

        # target = torch.zeros(1)
        # target = Variable(target.cuda())

        # # Sum of blindness class probs
        # sum_blindness_class_probs = blindness_class_probs.sum()
        #        sum_blindness_class_probs = blindness_class_logits.sum()

        # # L1Loss between sum of blindness class probs and zero tensor
        # loss = nn.L1Loss(size_average=True)(sum_blindness_class_probs, target)
        # #        max_class_prob, _ = torch.max(blindness_class_probs,0)
        # #        mean_class_prob = torch.mean(blindness_class_probs,0)


        # #        tmp11 = torch.reshape(cls_prob, (batch_size, -1, 20))
        # tmp11 = cls_prob.view(output.data.size(0), -1, 20)
        # tmp12 = tmp11[:, :, blindness_class_index]
        # tmp13 = tmp12.max(dim=1)[0].min(dim=0)[0]


        # -------------------------------------------------------------------------------------------------------------
        # VERSION 1
        # # target variable of zeros
        # target = torch.zeros(blindness_class_logits.size()[0], 1)
        # target = Variable(target.cuda())

        # # sum all logits
        # sum_logits = cls.sum(dim=1)
        # sum_logits = sum_logits.unsqueeze(1)

        # # calculate difference of logits
        # logits_diff = 2*blindness_class_logits - sum_logits

        # # print(sum_logits.size())
        # # if self.seen == 3:
        # #     pdb.set_trace()

        # # L1Loss between difference of logits and zero tensor
        # loss = nn.L1Loss(size_average=True)(logits_diff, target)
        # max_class_prob, _ = torch.max(blindness_class_probs,0)

        # -------------------------------------------------------------------------------------------------------------

        # VERSION 2
        # # target variable of zeros
        # target = torch.zeros(1)
        # target = Variable(target.cuda())

        # # sum all logits
        # sum_logits = cls.sum(dim=1)
        # sum_logits = sum_logits.unsqueeze(1)

        # # calculate difference of logits
        # logits_diff = 2*blindness_class_logits - sum_logits

        # # sum of logits difference
        # logits_diff_sum = logits_diff.sum()

        # # L1Loss between sum of logit difference and zero tensor
        # loss = nn.L1Loss(size_average=True)(logits_diff_sum, target)
        # max_class_prob, _ = torch.max(blindness_class_probs,0)
        # -------------------------------------------------------------------------------------------------------------

        # VERSION 3
        # # target variable of zeros
        # target = torch.zeros(blindness_class_probs.size()[0], 1)
        # target = Variable(target.cuda())

        # # sum all probabilities
        # sum_probs = cls_prob.sum(dim=1)
        # sum_probs = sum_probs.unsqueeze(1)

        # # calculate difference of probs
        # probs_diff = 2*blindness_class_probs - sum_probs

        # # L1Loss between difference of probs and zero tensor
        # loss = nn.L1Loss(size_average=True)(probs_diff, target)
        # max_class_prob, _ = torch.max(blindness_class_probs,0)
        # -------------------------------------------------------------------------------------------------------------

        # VERSION 4
        # # target variable of zeros
        # target = torch.zeros(1)
        # target = Variable(target.cuda())

        # # sum all probabilities
        # sum_probs = cls_prob.sum(dim=1)
        # sum_probs = sum_probs.unsqueeze(1)

        # # # calculate difference of probs
        # probs_diff = 2*blindness_class_probs - sum_probs

        # # sum of probs difference
        # probs_diff_sum = probs_diff.sum()

        # # L1Loss between sum of prob difference and zero tensor
        # loss = nn.L1Loss(size_average=True)(probs_diff_sum, target)
        # max_class_prob, _ = torch.max(blindness_class_probs,0)
        # -------------------------------------------------------------------------------------------------------------

        # VERSION 5 (ONE WE STARTED WITH)
        # target variable of zeros
        # target = torch.zeros(blindness_class_probs.size()[0], 1)
        # target = Variable(target.cuda())

        # # L1Loss between blindness class probs and zero tensor
        # loss = nn.L1Loss(size_average=True)(blindness_class_probs, target)
        # max_class_prob, _ = torch.max(blindness_class_probs,0)
        # -------------------------------------------------------------------------------------------------------------

        # # VERSION 6
        # target variable of zeros
        target = torch.zeros(1)
        target = Variable(target.cuda())

        # sum of blindness class probs
        sum_blindness_class_probs = blindness_class_probs.sum()

        # L1Loss between blindness class probs and zero tensor
        loss = nn.L1Loss(size_average=True)(sum_blindness_class_probs, target)
        max_class_prob, _ = torch.max(blindness_class_probs,0)
        # -------------------------------------------------------------------------------------------------------------

        t3 = time.time()

        return loss, max_class_prob.data[0]
#        return loss, tmp13.data[0]

class RegionLoss_TargetedAttack(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss_TargetedAttack, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0


    def forward(self, output, target, target_class_index):

        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors,\
                                                                         nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)



        cls_mask = (cls_mask == 1)

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())

        cls      = cls[cls_mask].view(-1, nC)
        cls_prob = F.softmax(cls)
        target_class_index_var = Variable(torch.LongTensor([target_class_index]).cuda())

        # gather logits and probabilities of target class
        target_class_logits = torch.gather(cls, 1, target_class_index_var.repeat(cls.size(0), 1))
        target_class_probs = torch.gather(cls_prob, 1, target_class_index_var.repeat(cls.size(0), 1))

        # -------------------------------------------------------------------------------------------------------------
        # VERSION 1
        # # target variable of zeros
        # target = torch.zeros(target_class_logits.size()[0], 1)
        # target = Variable(target.cuda())

        # # sum all logits
        # sum_logits = cls.sum(dim=1)
        # sum_logits = sum_logits.unsqueeze(1)

        # # calculate difference of logits
        # logits_diff = sum_logits - 2*target_class_logits

        # # print(sum_logits.size())
        # # if self.seen == 3:
        # #     pdb.set_trace()

        # # L1Loss between difference of logits and zero tensor
        # loss = nn.L1Loss(size_average=True)(logits_diff, target)
        # min_class_prob, _ = torch.min(target_class_probs,0)

        # -------------------------------------------------------------------------------------------------------------

        # # VERSION 6
        # target variable of zeros
        target = torch.FloatTensor([1.0*target_class_probs.size(0)])
        target = Variable(target.cuda())

        # Sum of target class probs
        sum_target_class_probs = target_class_probs.sum()

        # pdb.set_trace()
        # L1Loss between sum pf target class probs and one*number of detections tensor
        loss = nn.L1Loss(size_average=True)(sum_target_class_probs, target)
        min_class_prob, _ = torch.min(target_class_probs,0)
        # -------------------------------------------------------------------------------------------------------------

        t3 = time.time()

        return loss, min_class_prob.data[0]
