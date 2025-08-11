import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvLayer(nn.Module):
    """门控时间卷积层"""
    def __init__(self, in_channels, out_channels, time_kernel_size):
        super(TemporalConvLayer, self).__init__()
        # 使用padding来保持时间维度长度不变
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
        # x shape: (B, C, N, T), L_norm shape: (N, N)
        x = torch.einsum('bcnl,nm->bcml', x, L_norm)
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

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.residual_conv = None

    def forward(self, x, L_norm):
        residual = self.residual_conv(x) if self.residual_conv else x
        
        x = self.temporal_conv1(x)
        x = self.spatial_conv(x, L_norm)
        x = F.relu(x)
        x = self.temporal_conv2(x)

        # (B, C, N, T) -> (B, T, N, C) for LayerNorm and residual
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 2, 1)
        
        return x + residual

class STGCN(nn.Module):
    """STGCN主模型"""
    def __init__(self, config, L_norm):
        super(STGCN, self).__init__()
        self.register_buffer('L_norm', L_norm)
        
        num_nodes = config['data']['num_nodes']
        in_features = config['data']['num_features']
        num_timesteps_input = config['data']['num_timesteps_input']
        num_timesteps_output = config['data']['num_timesteps_output']
        
        out_channels = config['model']['out_channels']
        spatial_channels = config['model']['spatial_channels']
        time_kernel_size = config['model']['time_kernel_size']
        graph_kernel_size = config['model']['graph_kernel_size']

        self.st_conv_block1 = STConvBlock(in_features, spatial_channels, out_channels, time_kernel_size, graph_kernel_size, num_nodes)
        self.st_conv_block2 = STConvBlock(out_channels, spatial_channels, out_channels, time_kernel_size, graph_kernel_size, num_nodes)
        
        self.output_layer = nn.Conv2d(out_channels, num_timesteps_output, kernel_size=(1, num_timesteps_input))

    def forward(self, x):
        # x shape: (B, T_in, N, C_in) -> (B, C_in, N, T_in)
        x = x.permute(0, 3, 2, 1)
        
        x = self.st_conv_block1(x, self.L_norm)
        x = self.st_conv_block2(x, self.L_norm)
        
        #
        # ******** 这是核心的修改 ********
        #
        # self.output_layer(x) 的输出形状是 (B, T_out, N, 1)
        # 这已经符合我们期望的 (Batch, Time, Node, Feature) 格式
        # 我们需要将其 permute 以匹配 Y_batch 的 (B, T, N, C) 格式
        out = self.output_layer(x)  # Shape: (B, T_out, N, 1)
        
        # 将其维度调整为 (B, T_out, N, 1) 以匹配 Y_batch
        # permute from (B, C, H, W) to (B, H, C, W) - no, let's rethink
        # Conv2d output is (B, C_out, H_out, W_out) -> (B, T_out, N, 1)
        # Target shape is (B, T_out, N, 1)
        # So the output of the conv is ALREADY in the correct format!
        # The previous permute was swapping N and T. The new permute should re-order to match Y.
        # Target Y is (B, T_out, N, C_out=1)
        # Current 'out' is (B, C_out=T_out, H_out=N, W_out=1)
        # We need to map (B, T_out, N, 1) -> (B, T_out, N, 1)
        # It seems the shape is already correct! Let's permute it to be sure about the layout.
        # (B, C, H, W) -> (B, C, H, W)
        # Let's adjust the final output to match data loader format (B, T, N, C)
        out = out.permute(0, 1, 2, 3) # This does nothing, just for clarity
        # A better way is to make sure output format is (B, T_out, N, C_out)
        # Let's check the Y_batch format: (B, T_out, N, 1)
        # The conv output format: (B, T_out, N, 1)
        # They match. The error was the old permute.
        
        # Let's clean it up.
        out = self.output_layer(x) # Shape: (B, T_out, N, 1)
        
        # The output of conv2d is (B, C_out, H, W). Here it is (B, num_timesteps_output, num_nodes, 1).
        # This is already the correct shape and order. No permute is needed.
        # However, the input to the model is (B, T, N, C). Let's be consistent.
        # We want the output to be (B, T, N, C).
        # The conv output is (B, T_out, N, 1). This is exactly what we want.

        return out