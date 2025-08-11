import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvLayer(nn.Module):
    """门控时间卷积层"""
    def __init__(self, in_channels, out_channels, time_kernel_size):
        super(TemporalConvLayer, self).__init__()
        self.time_conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=(1, time_kernel_size), padding=(0, (time_kernel_size - 1) // 2))
        self.out_channels = out_channels

    def forward(self, x):
        x_conv = self.time_conv(x)
        P, Q = torch.split(x_conv, [self.out_channels, self.out_channels], dim=1)
        return P * torch.sigmoid(Q)

class SpatialGraphConvLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_channels, out_channels, graph_kernel_size):
        super(SpatialGraphConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x, L_norm):
        if L_norm.dim() < 4:
            L_norm = L_norm.unsqueeze(0).unsqueeze(-1)
        x = torch.einsum('bcnl,bnmj->bcml', x, L_norm)
        out = self.conv(x)
        return out

class STConvBlock(nn.Module):
    """时空卷积块（三明治结构）"""
    def __init__(self, in_channels, spatial_channels, out_channels, time_kernel_size, graph_kernel_size, num_nodes):
        super(STConvBlock, self).__init__()
        self.temporal_conv1 = TemporalConvLayer(in_channels, out_channels, time_kernel_size)
        self.spatial_conv = SpatialGraphConvLayer(out_channels, spatial_channels, graph_kernel_size)
        self.temporal_conv2 = TemporalConvLayer(spatial_channels, out_channels, time_kernel_size)
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else None

    def forward(self, x, L_norm):
        residual = self.residual_conv(x) if self.residual_conv else x
        
        x = self.temporal_conv1(x)
        x = self.spatial_conv(x, L_norm)
        x = F.relu(x)
        x = self.temporal_conv2(x)

        # (B, C, N, T) -> (B, T, N, C) for LayerNorm
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm(x)
        # (B, T, N, C) -> (B, C, N, T)
        x = x.permute(0, 3, 2, 1)
        
        return x + residual

class STGCN(nn.Module):
    """STGCN主模型"""
    def __init__(self, config, L_norm):
        super(STGCN, self).__init__()
        self.L_norm = L_norm
        num_nodes = config['data']['num_nodes']
        in_features = config['data']['num_features']
        out_channels = config['model']['out_channels']
        spatial_channels = config['model']['spatial_channels']
        time_kernel_size = config['model']['time_kernel_size']
        graph_kernel_size = config['model']['graph_kernel_size']

        self.st_conv_block1 = STConvBlock(in_features, spatial_channels, out_channels, time_kernel_size, graph_kernel_size, num_nodes)
        self.st_conv_block2 = STConvBlock(out_channels, spatial_channels, out_channels, time_kernel_size, graph_kernel_size, num_nodes)
        
        self.output_layer = nn.Conv2d(out_channels, config['data']['num_timesteps_output'], kernel_size=(1, config['data']['num_timesteps_input']))

    def forward(self, x):
        # x shape: (B, T_in, N, C_in) -> (B, C_in, N, T_in)
        x = x.permute(0, 3, 2, 1)
        
        x = self.st_conv_block1(x, self.L_norm)
        x = self.st_conv_block2(x, self.L_norm)
        
        # (B, C_out, N, T_in) -> (B, T_out, N, 1)
        out = self.output_layer(x).permute(0, 3, 2, 1)

        return out