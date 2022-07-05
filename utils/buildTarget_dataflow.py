import torch

torch.set_printoptions(precision=4, sci_mode=False)

targets = torch.tensor([[0.00000e+00, 3.00000e+00, 8.56075e-01, 2.81519e-01, 2.87847e-01, 3.28000e-01],
        [0.00000e+00, 6.00000e+01, 2.10291e-02, 7.19292e-01, 4.20581e-02, 7.44271e-02],
        [1.00000e+00, 2.30000e+01, 1.54860e-01, 1.06450e-01, 1.86765e-01, 2.12900e-01],
        [1.00000e+00, 3.00000e+00, 8.35184e-01, 2.22316e-01, 2.81199e-01, 1.79037e-01],
        [1.00000e+00, 2.10000e+01, 8.46906e-02, 5.73525e-01, 1.69381e-01, 2.93712e-01],
        [1.00000e+00, 3.10000e+01, 7.97368e-01, 5.69161e-01, 1.76462e-01, 4.13207e-01],
        [1.00000e+00, 3.10000e+01, 7.36355e-01, 5.52358e-01, 1.66935e-01, 4.12444e-01],
        [1.00000e+00, 3.10000e+01, 6.86220e-01, 5.49765e-01, 1.08373e-01, 3.35534e-01],
        [1.00000e+00, 3.10000e+01, 6.54444e-01, 5.53942e-01, 9.83106e-02, 3.33882e-01],
        [1.00000e+00, 3.10000e+01, 6.17032e-01, 5.43344e-01, 9.42992e-02, 3.26908e-01],
        [1.00000e+00, 0.00000e+00, 9.98244e-01, 5.58187e-01, 3.50819e-03, 2.04210e-02],
        [1.00000e+00, 3.10000e+01, 6.88386e-01, 5.66762e-01, 1.07415e-01, 3.36412e-01]])
anchors_group = torch.tensor([[[ 1.25000,  1.62500],
         [ 2.00000,  3.75000],
         [ 4.12500,  2.87500]],

        [[ 1.87500,  3.81250],
         [ 3.87500,  2.81250],
         [ 3.68750,  7.43750]],

        [[ 3.62500,  2.81250],
         [ 4.87500,  6.18750],
         [11.65625, 10.18750]]])
sizes = [[80,80,80,80],[40,40,40,40],[20,20,20,20]]
def build_targets( targets):


    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = 3, targets.shape[0]  # number of anchors, targets # (12, 6)
    print("这一批次的数据包含{}个目标".format(nt))
    print("数据的表头是：img_id, cls_id, x_norm, y_norm, w_norm, h_nor")
    print('the input targets shape: {}'.format(targets.shape))
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7)  # normalized to gridspace gain
    ai = torch.arange(na).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    print("所有{}个目标，要与{}个anchor进行一一比较，所以需要对targets进行变换".format(nt, na))
    print("targets变换后的shape：{}".format(targets.shape))

    g = 0.5  # bias

    # 偏移量：off 参考下图看
    # ----------|--------|--------|
    # |         | (0, -1)|        |
    # ----------|--------|--------|
    # | (-1, 0) | (0, 0) | (1, 0) |
    # ----------|--------|--------|
    # |         | (0, 1) |        |
    # ----------|--------|--------|
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    # 输出特征图的数量：nl
    print("target在不同的尺度特征图（grid）上处理")
    for i in range(3):

        print("在（{}，{}）特征图（网格）上处理".format(sizes[i][0],sizes[i][0]))
        anchors = anchors_group[i]
        print("获取{}特征图对应的anchors： {}".format(sizes[i][0],anchors))
        # p[i]：[n, c, h, w]
        # gain：[1, 1, w, h, w, h, 1]
        gain[2:6] = torch.tensor(sizes[i])  # xyxy gain
        # Match targets to anchors
        # targets: [img_id, cls_id, x_norm, y_norm, w_norm, h_norm, anchor_id]
        # 将标签中归一化后的xywh映射到特征图上
        print("变换前:")
        print(targets * torch.tensor([1,1,1,1,1,1,1]))
        t = targets * gain
        print("变换后:")
        print(t)

        if nt:
            # Matches
            # 获取anchor与gt的宽高比值，如果比值超出anchor_t，那么该anchor就会被舍弃，不参与loss计算
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1 / r).max(2)[0] < 4  # compare  # (r, 1 / r) 高宽比，gt/anchor, anchor/gt
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            print('满足要求的gt高宽与anchor的高宽比')
            print(j)
            t = t[j]  # filter
            print('最终保留适合锚框宽高的target')
            print(t)

            # Offsets
            # 中心点：gxy
            # 反转中心点：gxi
            gxy = t[:, 2:4]  # grid xy
            print("剩余目标中心点")
            print(gxy)
            gxi = gain[[2, 3]] - gxy  # inverse
            print("剩余目标的反转中心点")
            print(gxi)
            # 距离当前格子左上角较近的中心点，并且不是位于边缘格子内
            j, k = ((gxy % 1 < g) & (gxy > 1)).T # x, y
            # 距离当前格子右下角较近的中心点，并且不是位于边缘格子内
            # l, m = ((gxy % 1 > g) & (gxy > 1)).T
            l, m = ((gxi % 1 < g) & (gxi > 1)).T # xi, yi

            # j和l， k和m是互斥的，一个为True，那么另一个必定为False
            # j：(5, m)
            # j：[[all],
            #       [j == True],
            #       [k == True],
            #       [l == True],
            #       [m == True]]
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            print('不理解j是什么意思')
            print(j)
            # t：(5, m, 5)
            t = t.repeat((5, 1, 1)) # 3 * 7 ==> 5 * 3 * 7
            t = t[j]   # j = 5 * 3 其中有三行是true
            print('先重复再经过j的保留')
            print(t.shape)
            # shape：(1, m, 2) + (5, 1, 2) = (5, m, 2)[j] = (m', 2)
            # offsets排列(g = 0.5)：(0, 0), (0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5)
            print('处理offsets')
            offsets = (torch.zeros_like(gxy)[None] + off[:, None]) # 1 * 3 * 2 + 5 * 1 * 2 = 5 * 3 * 2
            print("未过滤前的offset")
            print(offsets)
            offsets = offsets[j]
            print('过滤后的offset')
            print(offsets)
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        # gxy和gwh当前是基于特征图尺寸的数据
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        print("ground true 中心点坐标")
        # 注意offsets的排列顺序，做减法，其结果为：(0, 0) + 四选二((-1, 0), (0, -1), (1, 0), (0, 1))
        # gij就是正样本格子的整数部分即索引
        gij = (gxy - offsets).long()
        print(gij)
        gi, gj = gij.T  # grid xy indices
        print(gi, gj)

        # Append
        # 去掉anchor_id
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        # 这里(gxy-gij)的取值范围-0.5 ~ 1.5
        print(torch.cat((gxy - gij, gwh), 1))

        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

build_targets(targets)