

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.ema import EMA
from nets.LWGANet import LWGANet_L2_1442_e96_k11_ReLU, LWGANet_L1_1242_e64_k11_ReLU, LWGANet_L0_1242_e32_k11_ReLU


class LWGANetBackbone(nn.Module):
    def __init__(self, model_type="l2", downsample_factor=8, pretrained=True):
        super(LWGANetBackbone, self).__init__()

        model_configs = {
            "l0": {
                "model_class": LWGANet_L0_1242_e32_k11_ReLU,
                "embed_dim": 32,
                "dropout": 0.0,
                "pretrained_path": 'model_data/lwganet_l0_e299.pth',
                "channels": [32, 64, 128, 256]
            },
            "l1": {
                "model_class": LWGANet_L1_1242_e64_k11_ReLU,
                "embed_dim": 64,
                "dropout": 0.1,
                "pretrained_path": 'model_data/lwganet_l1_e299.pth',
                "channels": [64, 128, 256, 512]
            },
            "l2": {
                "model_class": LWGANet_L2_1442_e96_k11_ReLU,
                "embed_dim": 96,
                "dropout": 0.1,
                "pretrained_path": 'model_data/lwganet_l2_e296.pth',
                "channels": [96, 192, 384, 768]
            }
        }

        if model_type not in model_configs:
            raise ValueError(f"Unsupported model_type: {model_type}. Use l0, l1, or l2.")

        config = model_configs[model_type]
        model = config["model_class"](
            embed_dim=config["embed_dim"],
            dropout=config["dropout"],
            fork_feat=True
        )

        if pretrained:
            try:
                checkpoint = torch.load(config["pretrained_path"], map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("module.", "")
                    new_state_dict[name] = v

                model.load_state_dict(new_state_dict, strict=False)
                print(f"Successfully loaded pretrained weights for {model_type}")
            except Exception as e:
                print(f"Failed to load pretrained weights for {model_type}: {e}")
                print("Training from scratch...")

        self.features = model
        self.downsample_factor = downsample_factor
        self.stage_channels = config["channels"]

        self.stage1_channels = self.stage_channels[0]
        self.stage2_channels = self.stage_channels[1]
        self.stage3_channels = self.stage_channels[2]
        self.stage4_channels = self.stage_channels[3]

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.features(x)
        stage1_features = features[0]  
        stage2_features = features[1]  
        stage3_features = features[2]  
        stage4_features = features[3]  
        return stage1_features, stage2_features, stage3_features, stage4_features


class Neck_LGA_DC_EMA(nn.Module):
    def __init__(self, stage_channels, neck_channels=256):
        super(Neck_LGA_DC_EMA, self).__init__()

        
        self.stage1_ch = stage_channels[0]
        self.stage2_ch = stage_channels[1]
        self.stage3_ch = stage_channels[2]
        self.stage4_ch = stage_channels[3]
        self.neck_ch = neck_channels

        self.stage4_conv = nn.Sequential(
            nn.Conv2d(self.stage4_ch, self.neck_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(self.neck_ch),
            nn.ReLU(inplace=True),
        )

        self.dc_block3 = nn.Sequential(
            nn.Conv2d(self.neck_ch + self.stage3_ch, self.neck_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(self.neck_ch),
            nn.ReLU(inplace=True),
        )

        self.dc_block2 = nn.Sequential(
            nn.Conv2d(self.neck_ch + self.stage2_ch, self.neck_ch, 3, padding=4, dilation=4),
            nn.BatchNorm2d(self.neck_ch),
            nn.ReLU(inplace=True),
        )

        self.dc_block1 = nn.Sequential(
            nn.Conv2d(self.neck_ch + self.stage1_ch, self.neck_ch, 3, padding=8, dilation=8),
            nn.BatchNorm2d(self.neck_ch),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.neck_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ema = EMA(64)

    def forward(self, stage1, stage2, stage3, stage4):
        
        x = self.stage4_conv(stage4)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, stage3], dim=1)
        x = self.dc_block3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, stage2], dim=1)
        x = self.dc_block2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, stage1], dim=1)
        x = self.dc_block1(x)

        x = self.final_conv(x)

        x = self.ema(x)

        return x


class LGASSNet(nn.Module):
    def __init__(self, num_classes, backbone="lwganet_l2", pretrained=True, downsample_factor=16):
        super(LGASSNet, self).__init__()

        self.backbone_name = backbone
        self.use_lga_dc_ema = False

        if backbone in ["lwganet_l0", "lwganet_l1", "lwganet_l2", "lwganet"]:
            model_type = backbone.replace("lwganet_", "") if backbone != "lwganet" else "l2"
            self.backbone = LWGANetBackbone(
                model_type=model_type,
                downsample_factor=downsample_factor,
                pretrained=pretrained
            )

            
            self.use_lga_dc_ema = True
            stage_channels = self.backbone.stage_channels

            
            self.neck_lga_dc_ema = Neck_LGA_DC_EMA(stage_channels, neck_channels=256)

           
            self.cls_conv = nn.Conv2d(64, num_classes, 1)

            return

        else:
            raise ValueError(f'Unsupported backbone - `{backbone}`, Use mobilenet, xception, lwganet.')


    def forward(self, x):
        H, W = x.size(2), x.size(3)

        if self.use_lga_dc_ema:
            stage1, stage2, stage3, stage4 = self.backbone(x)
            x = self.neck_lga_dc_ema(stage1, stage2, stage3, stage4)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
            x = self.cls_conv(x)

        return x

    def freeze_backbone(self):
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = True


class LWGANetBackbone_l0(LWGANetBackbone):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__(model_type="l0", downsample_factor=downsample_factor, pretrained=pretrained)


class LWGANetBackbone_l1(LWGANetBackbone):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__(model_type="l1", downsample_factor=downsample_factor, pretrained=pretrained)


class LWGANetBackbone_l2(LWGANetBackbone):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__(model_type="l2", downsample_factor=downsample_factor, pretrained=pretrained)



if __name__ == '__main__':

    backbones = ["lwganet_l0", "lwganet_l1", "lwganet_l2"]

    for backbone in backbones:
        print(f"\n=== Testing {backbone} ===")
        x = torch.randn((2, 3, 512, 512))
        net = LGASSNet(num_classes=4, backbone=backbone, pretrained=False, downsample_factor=16)
        features = net(x)
        print(f"Output shape: {features.shape}")




    