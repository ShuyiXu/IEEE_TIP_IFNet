import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .modules.resnet import *
from .modules.trans import Transformer, TransformerDecoder, TwoLayerConv2d
from .modules.GatedSpatialConv import GatedSpatialConv2d

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits):
        y = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard)
        return y

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        # 获取输入的波段数
        num_bands = x.size(1)
        
        # 初始化一个列表来存储每个波段的梯度结果
        gradient_bands = []
        
        # 对每个波段计算梯度
        for i in range(num_bands):
            xi = x[:, i].unsqueeze(1)  # 提取第i个波段并增加一个通道维度
            xi_v = F.conv2d(xi, self.weight_v, padding=1)  # 计算垂直梯度
            xi_h = F.conv2d(xi, self.weight_h, padding=1)  # 计算水平梯度
            
            # 计算梯度幅度
            gradient = torch.sqrt(torch.pow(xi_v, 2) + torch.pow(xi_h, 2) + 1e-6)
            gradient_bands.append(gradient)
        
        # 将所有波段的梯度结果合并回一个张量
        x = torch.cat(gradient_bands, dim=1)
        return x

class TFD(nn.Module):
    def __init__(self, inch, outch):
        super(TFD, self).__init__()
        self.res1 = BasicBlock1(inch, outch, stride=1, downsample=None)
        self.res2 = BasicBlock1(inch, outch, stride=1, downsample=None)
        self.gate = GatedSpatialConv2d(inch, outch)
    def forward(self,x,f_x):#coarse,fm
        u_0 = x
        u_1, delta_u_0 = self.res1(u_0)
        _, u_2 = self.res2(u_1)
        u_3_pre = self.gate(u_2, f_x)
        u_3 = 3 * delta_u_0 + u_2 + u_3_pre
        return u_3

class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet =resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class IFNet(ResNet):

    def __init__(self, input_nc1,input_nc2,select_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(IFNet, self).__init__(select_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim
        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)
        self.gumbel = GumbelSoftmax()
        self.input_nc1 = input_nc1
        self.input_nc2 = input_nc2
        self.select_nc = select_nc
        # 初始化Gumbel Softmax选择器
        self.band_weights1 = nn.Parameter(torch.randn(1, input_nc1))  # Shape [1, input_nbr1] for broadcasting
        self.band_weights2 = nn.Parameter(torch.randn(1, input_nc2))  # Shape [1, input_nbr2] for broadcasting
        self.getgrad = Get_gradient_nopadding()
        self.tfd = TFD(32,32)
        self.dsn = nn.Conv2d(32, 1, 1)
        self.fuse = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        _, _, hei, wid = x1.shape
        # x_size = x1.size()
        batch_size = x1.size(0)

        band_weights1 = self.gumbel(self.band_weights1.expand(batch_size, -1))
        band_weights2 = self.gumbel(self.band_weights2.expand(batch_size, -1))
        # 获取每个样本权重最高的 select_nc 个波段索引
        _, selected_band_indices1 = torch.topk(band_weights1, self.select_nc, dim=1)  # 形状: [batch_size, select_nc]
        _, selected_band_indices2 = torch.topk(band_weights2, self.select_nc, dim=1)  # 形状: [batch_size, select_nc]
        # 将选定波段索引扩展到高度和宽度维度，使其形状与 x1 和 x2 匹配
        selected_band_indices1 = selected_band_indices1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x1.size(2),
                                                                                           x1.size(3))
        selected_band_indices2 = selected_band_indices2.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x2.size(2),
                                                                                           x2.size(3))

        # 使用 gather 直接选择每个样本的 select_nc 个波段
        x1_selected = torch.gather(x1, 1, selected_band_indices1)  # 形状: [batch_size, select_nc, height, width]
        x2_selected = torch.gather(x2, 1, selected_band_indices2)  # 形状: [batch_size, select_nc, height, width]
        # print(f'Selected bands for x1: {selected_band_indices1[:, :, 0, 0]}')  # Prints selected band indices
        # print(f'Selected bands for x2: {selected_band_indices2[:, :, 0, 0]}')
        x1 = x1_selected
        x2 = x2_selected

        x1 = self.forward_single(x1)#32,64,64
        x2 = self.forward_single(x2)
        coarse_edge = self.getgrad(torch.abs(x1-x2))
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)#4,32,32
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)#4,32,64,64
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)# 4,32,64,64 精细特征
        xdsn = self.dsn(x)
        refine_edge = self.tfd(coarse_edge,xdsn)
        refine_edge = self.fuse(refine_edge)

        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        refine_edge = self.upsamplex4(refine_edge)
        edge_out = self.sigmoid(refine_edge)
        x = x * edge_out + x
        # forward small cnn
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x, edge_out






