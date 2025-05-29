"""
Re-implemented forward function of image matching by FYL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np
import cv2

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.gate = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        # return self.gate(self.bn2(self.conv2(self.gate(self.bn1(self.conv1(x))))))
        y = self.conv1(x)  # cn: 1->16
        y = self.bn1(y)
        y = self.gate(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.gate(y)
        return y

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.gate = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(inplanes, planes, 1, bias=True)
    def forward(self, x):
        # return self.gate(self.bn2(self.conv2(self.gate(self.bn1(self.conv1(x))))) + self.conv3(x))
        y = self.conv1(x)  # cn: 16->32
        y = self.bn1(y)
        y = self.gate(y)
        y = self.conv2(y)
        y = self.bn2(y)
        z = self.conv3(x)  # cn: 16->32
        y = y + z
        y = self.gate(y)
        return y


class fylmBackbone(nn.Module):
    """ FYL matcher backbone model """
    def __init__(self):
        super(fylmBackbone, self).__init__()

        self.gate = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        c1, c2, c3, c4, c5 = 16, 32, 64, 128, 256
        self.block1 = ConvBlock(1, c1)
        self.block2 = ResBlock(inplanes=c1, planes=c2)
        self.block3 = ResBlock(inplanes=c2, planes=c3)
        self.block4 = ResBlock(inplanes=c3, planes=c4)
        self.block5 = ResBlock(inplanes=c4, planes=c5)
        block_dims = [32, 64, 128, 256]
        self.conv4 = nn.Conv2d(block_dims[3], block_dims[3], 1, bias=False)
        self.conv3 = nn.Conv2d(block_dims[2], block_dims[3], 1, bias=False)
        self.conv2 = nn.Conv2d(block_dims[1], block_dims[2], 1, bias=False)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(block_dims[3], block_dims[3], 3, bias=False, padding=1),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            nn.Conv2d(block_dims[3], block_dims[2], 3, bias=False, padding=1),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(block_dims[2], block_dims[2], 3, bias=False, padding=1),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            nn.Conv2d(block_dims[2], block_dims[2], 3, bias=False, padding=1),
        )

    def forward(self, x):
        # x(size=1,cn=1)
        y1 = self.block1(x)  # (1,16)
        y2 = self.pool2(y1)  # (1/2,16)
        y3 = self.block2(y2)  # (1/2,32)
        y4 = self.pool2(y3)  # (1/4,32)
        y5 = self.block3(y4)  # (1/4,64)
        y6 = self.pool2(y5)  # (1/8,64)
        y7 = self.block4(y6)  # (1/8,128)
        y8 = self.pool2(y7)  # (1/16,128)
        y9 = self.block5(y8)  # (1/16,256)
        y9 = self.conv4(y9)  # (1/16,256)
        z1 = tf.interpolate(y9, y7.shape[2:], mode='bilinear', align_corners=True)  # (1/8,256)
        z2 = self.conv3(y7)  # (1/8,256)
        z1 = z1 + z2
        y7 = self.mlp1(z1)  # (1/8,128)
        z3 = tf.interpolate(y7, y5.shape[2:], mode='bilinear', align_corners=True)  # (1/4,128)
        z4 = self.conv2(y5)  # (1/4,128)
        z3 = z3 + z4  # (1/4,128)
        z3 = self.mlp2(z3)  # (1/4,128)
        z3 = tf.normalize(z3, dim=1)  # (1/4,128)
        return z3
class fylmHead(nn.Module):
    def __init__(self):
        super(fylmHead, self).__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 1, 1, 1, 0)
    def forward(self, x):
        # x(size=1,cn=1)
        y = self.conv(x)  # (1,16)
        y = tf.instance_norm(y)  # (1,16); nn.InstanceNorm2d(16)
        x = self.conv2(y)  # (1,1)
        x = tf.instance_norm(x)  # (1,1); nn.InstanceNorm2d(1)
        x = tf.sigmoid(x)  # (1,1)
        return x

def extract_keypoints(heat, radius=2, min_thresh=0.2, max_keypoints=1024):
    assert min_thresh > 1e-5
    mask = heat[radius:-radius, radius:-radius] > min_thresh
    l = 2 * radius + 1
    mm = tf.max_pool2d(heat[None, None], kernel_size=l, stride=1, padding=0).squeeze()
    peak = mm == heat[radius:-radius, radius:-radius]
    mask = peak & mask
    mm[~mask] = 0
    topk = mm.flatten().topk(k=max_keypoints, largest=True, sorted=False)  # (maxk)
    scores = topk.values  # (max_keypoints), float32
    y = (topk.indices // mm.shape[-1]) + radius  # (max_keypoints), int64
    x = (topk.indices % mm.shape[-1]) + radius  # (max_keypoints), int64; to heat's coord

    # Differentiable keypoint detection
    neighbors = torch.arange(-radius, radius + 1)  # (l), int64
    xx = x.reshape(-1, 1, 1) + neighbors.reshape(1, 1, -1)  # (maxk,l), int64
    yy = y.reshape(-1, 1, 1) + neighbors.reshape(1, -1, 1)  # (maxk,l), int64
    patch_scores = heat[yy, xx]  # (maxk,l,l), float32
    assert torch.allclose(patch_scores[:, radius, radius], scores)
    patch_scores = (patch_scores - scores.reshape(-1, 1, 1)) / 0.1  # (maxk,l,L), float32
    patch_scores = patch_scores.flatten(1)  # (maxk,l*l)
    patch_scores = patch_scores.softmax(dim=1)  # (maxk,l*l)
    patch_scores = patch_scores.unflatten(-1, [l, l])  # (maxk,l,l)
    neighbors = neighbors.to(dtype=torch.float32)  # (l)
    xbar = (patch_scores * neighbors.reshape(1, 1, -1)).sum(dim=[1, 2])  # (maxk)
    ybar = (patch_scores * neighbors.reshape(1, -1, 1)).sum(dim=[1, 2])  # (maxk)

    y = y.to(dtype=torch.float32) + ybar  # (maxk)
    x = x.to(dtype=torch.float32) + xbar  # (maxk)
    return scores, x, y  # keypoint scores and keypoint positions

class fylm(nn.Module):
    """ FYL matcher """
    def __init__(self, weight_file: str=None):
        super(fylm, self).__init__()
        self.backbone1 = fylmBackbone()
        self.backbone2 = fylmBackbone()
        self.head = fylmHead()
        if weight_file is not None:
            self.load(weight_file)
    def forward(self, gray, max_keypoints=1024):
        # gray : (h,w) shaped gray image, uint8, np.ndarray
        # cv2.imshow('img', gray)
        heat = torch.tensor(gray, dtype=torch.float32)[None, None] / 255  # (1,1,h,w)
        heat = tf.instance_norm(heat)  # (size=1,cn=1)
        F1 = self.backbone1(heat)  # (1/4,128)
        F2 = self.backbone2(heat)  # (1/4,128)
        heat = self.head(heat)  # (size=1,cn=1)

        scores, x, y = extract_keypoints(heat[0, 0], max_keypoints=max_keypoints)
        h, w = heat.shape[-2:]
        xn = x * (2 / (w - 1)) - 1
        yn = y * (2 / (h - 1)) - 1
        points = torch.stack([xn, yn], dim=-1)  # (1024,2)
        _f1 = tf.grid_sample(F1, points[None, None], align_corners=True).squeeze()  # (1,128,1,1024)->(128,1024)
        _f2 = tf.grid_sample(F2, points[None, None], align_corners=True).squeeze()  # (1,128,1,1024)->(128,1024)
        f1 = tf.normalize(_f1.T, dim=1)  # (1024,128)
        f2 = tf.normalize(_f2.T, dim=1)  # (1024,128)
        return f1, f2, x, y

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath), strict=True)

def match_mnn(f11, f12, f21, f22, thresh=0.8):  # f(D,N)
    N1, N2 = f11.shape[0], f21.shape[0]
    # assert N1 == N2
    sim = f11 @ f21.T  # (N1,N2), where N1=N2
    sim2 = f12 @ f22.T  # (N1,N2)
    weight = 0.25
    sim = sim * weight + sim2 * (1 - weight)
    nn12 = sim.argmax(dim=1)  # (N1), nearest p2 for each p1
    nn21 = sim.argmax(dim=0)  # (N2), nearest p1 for each p2
    idx = torch.arange(N1)
    mask = idx == nn21[nn12]
    scores = sim[idx, nn12]  # (N1)
    mask2 = scores > thresh
    mask = mask & mask2
    return mask, nn12  # mask(N):bool, nn12(N):int64

def draw_matches(im1, im2, x1, y1, x2, y2, mask, nn12, max_matches=100):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    dx, dy = 30, 120
    H, W = max(h1, h2) + dy, w1 + w2 + dx
    mosaic = np.zeros([H, W, 3], dtype=np.uint8)
    mosaic[:h1, :w1] = im1
    mosaic[-h2:, -w2:] = im2
    # Draw keypoints
    if len(x1):
        x1 = x1.round().to(dtype=torch.int32).numpy()
        y1 = y1.round().to(dtype=torch.int32).numpy()
        for x, y in zip(x1, y1):
            a, b, c, d = max(0, x - 2), max(0, y - 2), min(w1, x + 3), min(h1, y + 3)
            mosaic[b : d, a : c, 2] = 255
    dx, dy = dx + w1, H - h2
    if len(x2):
        x2 = x2.round().to(dtype=torch.int32).numpy()
        y2 = y2.round().to(dtype=torch.int32).numpy()
        for x, y in zip(x2, y2):
            a, b, c, d = max(0, x - 2), max(0, y - 2), min(w2, x + 3), min(h2, y + 3)
            a, b, c, d = a + dx, b + dy, c + dx, d + dy
            mosaic[b : d, a : c, 1] = 255
    # Draw matches
    idx1 = torch.where(mask)[0].numpy()
    if len(idx1):
        idx2 = nn12.numpy()[idx1]
        x1, y1, x2, y2 = x1[idx1], y1[idx1], x2[idx2] + dx, y2[idx2] + dy
        colors = [
            [0, 0, 192],  # dark red
            [0, 0, 255],  # red
            [0, 192, 255],  # earth yellow
            [0, 255, 255],  # yellow
            [80, 208, 146],  # cyan
            [80, 176, 0],  # deep green
            [240, 176, 0],  # deep blue
            [192, 112, 0],  # dark blue
            [96, 32, 0],  # darker blue
            [160, 48, 112],  # purple
            [213, 155, 91],  # cornflower blue
            [106, 84, 68],  # slate gray
            [17, 90, 197],  # chocolate
            [124, 124, 124],  # gray
            [237, 237, 237],  # light gray
            [153, 230, 255],  # light orange
            [217, 240, 226],  # lime
        ]
        ii = np.arange(len(idx1))
        if max_matches is not None:
            np.random.shuffle(ii)
            ii = ii[:min(max_matches, len(ii))]
        for i in ii:
            cv2.line(mosaic, pt1=(x1[i], y1[i]), pt2=(x2[i], y2[i]), thickness=3, color=colors[i % len(colors)])
    # cv2.imshow('matches', mosaic)
    cv2.imwrite("1.jpg", mosaic)
    # cv2.waitKey()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fm = fylm('fylm.pth')
    im1 = cv2.imread(r"D:\python mode\DeDoDe-main\assets\im_A.jpg")
    im2 = cv2.imread(r"D:\python mode\DeDoDe-main\assets\im_B.jpg")
    # im1 = cv2.resize(im1, dsize=(640, 480), fx=0, fy=0)
    # im2 = cv2.resize(im2, dsize=(640, 480), fx=0, fy=0)
    # im1 = cv2.pyrDown(im1)
    # im2 = cv2.pyrDown(im2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    max_keypoints, match_thresh = 1024, 0.8
    f11, f12, x1, y1 = fm(gray1, max_keypoints)
    f21, f22, x2, y2 = fm(gray2, max_keypoints)
    mask, nn12 = match_mnn(f11, f12, f21, f22, match_thresh)
    draw_matches(im1, im2, x1, y1, x2, y2, mask, nn12)

