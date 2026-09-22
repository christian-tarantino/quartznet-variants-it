from torch import nn
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T





class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, batch_first=False):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation=nn.SiLU(),
            batch_first=batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)







class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2  # Calcola il padding per mantenere la dimensione
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x, activate=True):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if activate:
            x = self.act(x)

        return x

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1):
        super().__init__()
        self.pres = nn.Conv1d(in_channels, out_channels, kernel_size=1,padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,)        
        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=kernel_size,stride=stride,)
        self.conv3 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,)        
        self.conv4 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=kernel_size,stride=stride,)
        self.conv5 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,)        
        self.relu = nn.SiLU()

    def forward(self, x):
        res = self.pres(x)
        res = self.bn(res)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out, activate=False)
        out = out + res 
        out = self.relu(out)      
        return out 


class QuartzNet_evo_2(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()

        self.input_norm = nn.InstanceNorm1d(80, affine=True)

        self.conv1 = DepthwiseSeparableConv1d(80, 256, 33, stride=2)
        self.attention_block1 = TransformerBlock(256,8,1, True)

        self.b1s1 = ResidualBlock(256,256,33,)
        self.b2s1 = ResidualBlock(256,256,39,)
        self.b3s1 = ResidualBlock(256,512,51,)
        self.b4s1 = ResidualBlock(512,512,63,)
        self.b5s1 = ResidualBlock(512,512,75,)
        self.b1s2 = ResidualBlock(256,256,33,)
        self.b2s2 = ResidualBlock(256,256,39,)
        self.b3s2 = ResidualBlock(512,512,51,)

        self.conv2 = DepthwiseSeparableConv1d(512,512,87,)
        self.attention_block2=TransformerBlock(512,8,1)
        self.fc = nn.Linear(512, num_classes)


        self._init_weights()

    def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv1d):  
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm1d, nn.LayerNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, input_lengths):
        x = self.input_norm(x)  # Normalize

        x = self.conv1(x)
        x = x.permute(0,2,1) # to match attention block format
        x = self.attention_block1(x)
        x = x.permute(0,2,1)

        x = self.b1s1(x)
        x = self.b1s2(x)
        

        x = self.b2s1(x)
        x = self.b2s2(x)
        

        x = self.b3s1(x)
        x = self.b3s2(x)
        

        x = self.b4s1(x)
        
        x = self.b5s1(x)

        x = self.conv2(x)
        x = x.permute(2,0,1)
        x = self.attention_block2(x)
        logits = self.fc(x)

        output_lengths = input_lengths.flatten()
        # rounding output lenghts 
        output_lengths = torch.div(output_lengths - 1, 2, rounding_mode='floor') + 1

        return logits, output_lengths