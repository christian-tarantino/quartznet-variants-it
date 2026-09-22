from torch import nn
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
'''embed_dim
Total dimension of the model.

num_heads
Number of parallel attention heads. Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).

dropout
Dropout probability on attn_output_weights. Default: 0.0 (no dropout).

bias
If specified, adds bias to input / output projection layers. Default: True.

add_bias_kv
If specified, adds bias to the key and value sequences at dim=0. Default: False.

add_zero_attn
If specified, adds a new batch of zeros to the key and value sequences at dim=1. Default: False.

kdim
Total number of features for keys. Default: None (uses kdim=embed_dim).

vdim
Total number of features for values. Default: None (uses vdim=embed_dim).

batch_first'''





    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, num_layers, batch_first=False):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation=nn.SiLU(),
            batch_first=batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x arriva nel formato [Time, Batch, Channels] (se batch_first=False)
        return self.transformer(x)







class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()

        # 1. Depthwise: Estrazione temporale/spaziale
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.gn1 = nn.GroupNorm(1, in_channels)
        self.act1 = nn.SiLU()

        # 2. Pointwise: Proiezione/Mix dei canali
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,padding=0, bias=False)
        self.gn2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.act1(self.gn1(self.depthwise(x)))
        x = self.act2(self.gn2(self.pointwise(x)))
        return x









    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2  # Calcola il padding per mantenere la dimensione
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
                nn.GroupNorm(1, out_channels)
            )
        self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout = nn.Dropout1d(p=0.15)
        
        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=kernel_size,stride=1, padding=padding)
    def forward(self, x):
        res = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Somma lineare (Identity shortcut)
        return out + res


class ASRResNet_Medium(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()

        self.input_norm = nn.InstanceNorm1d(80, affine=True)

        self.attn=TransformerBlock(embed_dim=784, num_heads=8, dropout=0.15, num_layers=1)


        self.layer1 = nn.Sequential(
            ResidualBlock(80, 80, kernel_size=9, stride=1),
            ResidualBlock(80, 80, kernel_size=9, stride=1),
            ResidualBlock(80, 80, kernel_size=9, stride=1),
        )

        # 2. Secondo Dimezzamento
        self.layer2 = nn.Sequential(
            ResidualBlock(80, 160, kernel_size=11, stride=1),  # Mantieni la stessa dimensione temporale
            ResidualBlock(160, 160, kernel_size=11, stride=1),
            ResidualBlock(160, 160, kernel_size=11, stride=1),
        )

        # Nessun altro dimezzamento temporale
        self.layer3 = nn.Sequential(
            ResidualBlock(160, 256, kernel_size=13, stride=2), 
            ResidualBlock(256, 256, kernel_size=13, stride=1),
            ResidualBlock(256, 256, kernel_size=13, stride=1)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, kernel_size=15, stride=2),
            ResidualBlock(512, 512, kernel_size=15, stride=1),
            ResidualBlock(512, 512, kernel_size=15, stride=1)
        )
        self.layer5 = nn.Sequential(
            ResidualBlock(512, 784, kernel_size=17, stride=1),
            ResidualBlock(784, 784, kernel_size=17, stride=1),
            ResidualBlock(784, 784, kernel_size=17, stride=1)
        )

        self.fc = nn.Linear(784, num_classes)
        self.fc_dropout = nn.Dropout1d(p=0.2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)       

    def forward(self, x, input_lengths):
        x = self.input_norm(x)  # Normalizza ogni spettrogramma in ingresso
        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x)  # Subsampling 3: Time / 2
        x = self.layer4(x)  # Subsampling 4: Time / 2
        x = self.layer5(x)  # Subsampling 5: Time / 2

        x = x.permute(2, 0, 1)  
       
        x = self.attn(x)
        x = self.fc_dropout(x)
        logits = self.fc(x)     # -> [Time, Batch, Num_Classes]

        output_lengths = input_lengths.flatten()

        for _ in range(2):
            output_lengths = torch.div(output_lengths - 1, 2, rounding_mode='floor') + 1

        return logits, output_lengths