import mindspore
import math
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

class SelfAttentionBlock2D(nn.Cell): # 迁移完成
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.SequentialCell(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
        )
        self.f_query = nn.SequentialCell(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
        )

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, has_bias=False)
        self.W = nn.SequentialCell(
            nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
                  

    def construct(self, x):
        batch_size, h, w = x.shape(0), x.shape(2), x.shape(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = ops.Transpose(value, (0, 2, 1))
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = ops.Transpose(query, (0, 2, 1))
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = nn.MatMul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = ops.Softmax(sim_map, dim=-1)

        context = nn.MatMul(sim_map, value)
        context = ops.Reshape(ops.Transpose(context, (0, 2, 1)), (batch_size, self.value_channels, h, w)) # torch.reshape() 大致相当于tensor.contiguous().view()
        context = self.W(context)
        return context


class ISA_Block(nn.Cell): # 迁移完成
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factor=[8,8], bn_type=None):
        super(ISA_Block, self).__init__()
        self.out_channels = out_channels
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        self.long_range_sa = SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
        self.short_range_sa = SelfAttentionBlock2D(out_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
    
    def construct(self, x):
        n, c, h, w = x.shape()
        dh, dw = self.down_factor       # down_factor for h and w, respectively
        
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        # pad the feature if the size is not divisible
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:  # padding in both left&right sides
            pad_op = nn.Pad(paddings=(pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
            feats = pad_op(x)
        else:
            feats = x
        
        # long range attention
        feats = feats.view(n, c, out_h, dh, out_w, dw)
        #feats = feats.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)
        feats = ops.Reshape(ops.Transpose(feats, (0, 3, 5, 1, 2, 4)), (-1, c, out_h, out_w))
        feats = self.long_range_sa(feats)
        c = self.out_channels

        # short range attention
        feats = feats.view(n, dh, dw, c, out_h, out_w)
        #feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw)
        feats = ops.Reshape(ops.Transpose(feats, (0, 4, 5, 3, 1, 2)), (-1, c, dh, dw))
        feats = self.short_range_sa(feats)
        #feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = ops.Transpose(feats.view(n, out_h, out_w, c, dh, dw), (0, 3, 1, 4, 2, 5))
        #feats = feats.contiguous().view(n, c, dh * out_h, dw * out_w)
        feats = ops.Reshape(feats, (n, c, dh * out_h, dw * out_w))

        # remove padding
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]
        
        return feats


class ISA_Module(nn.Cell): # 迁移完毕
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factors=[[8,8]], dropout=0, bn_type=None):
        super(ISA_Module, self).__init__()

        assert isinstance(down_factors, (tuple, list))
        self.down_factors = down_factors

        self.stages = nn.CellList([
            ISA_Block(in_channels, key_channels, value_channels, out_channels, d, bn_type) for d in down_factors
        ])

        concat_channels = in_channels + out_channels
        if len(self.down_factors) > 1:
            self.up_conv = nn.SequentialCell(
                nn.Conv2d(in_channels=in_channels, out_channels=len(self.down_factors) * out_channels, kernel_size=1, padding=0, has_bias=False),
                nn.BatchNorm2d(len(self.down_factors) * out_channels),
                nn.ReLU(),
            )
            concat_channels = out_channels * len(self.down_factors) * 2
        
        self.conv_bn = nn.SequentialCell(
            nn.Conv2d(in_channels=concat_channels, out_channels=out_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(float(1 - dropout)),
        )
    
    def construct(self, x):
        priors = [stage(x) for stage in self.stages]
        if len(self.down_factors) == 1:
            context = priors[0]
        else:
            #context = torch.cat(priors, dim=1)
            context = ops.Concat(axis=1)(priors)
            x = self.up_conv(x)
        # residual connection
        #return self.conv_bn(torch.cat([x, context], dim=1))
        return self.conv_bn(ops.Concat(axis=1)([x, context]))


if __name__ == "__main__":
    #feats = torch.randn((1, 1024, 56, 56))
    feats = ops.StandardNormal()((1, 1024, 56, 56))
    model = ISA_Module(in_channels=1024, key_channels=256, value_channels=512, out_channels=512, dropout=0)
    out = model(feats)
    print(out.shape)

# torch.Size([1, 512, 56, 56])