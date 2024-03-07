import sys, os
sys.path.append(os.pardir)
from common.util import im2col, col2im
import numpy as np

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        # 各種データのサイズを取得
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        # 入力された4次元データを行列に変換
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 重みをいい感じに行列に変換
        # -1とすることで辻褄が合うように自動で大きさを調整してくれる
        col_W = self.W.reshape(FN, -1).T
        # フィルターを掛ける
        out = np.dot(col, col_W) + self.b
        # データを4次元配列に戻す
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        # フィルターのサイズを取得
        FN, C, FH, FW = self.W.shape
        # doutの形を整形
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        # 各パラメータの微分を求める
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).rehape(FN, C, FH, FW)
        # 逆伝播の出力を行列形式で計算
        dcol = np.dot(dout, self.col_W.T)
        # 出力を4次元配列に戻す
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


