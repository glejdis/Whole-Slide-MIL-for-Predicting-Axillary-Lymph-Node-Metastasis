from torch import nn
import torchvision
from .googlenet import googlenet
from .inception import inception_v3

BACKBONES = [
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "googlenet",
    "inception_v3",
    "alexnet",
    'mobilenet_v2',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
    "efficientnet_v2_s",
    'efficientnet_v2_m',
    'efficientnet_v2_l',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'squeezenet1_0',
    'squeezenet1_1',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'resnext50_32x4d',
    'resnext101_32x8d',
]


class BackboneBuilder(nn.Module):
    """Build backbone with the last fc layer removed"""

    def __init__(self, backbone_name):
        super().__init__()

        assert backbone_name in BACKBONES

        if "googlenet" in backbone_name or "inception_v3" in backbone_name:  # These backbone has multiple output
            complete_backbone = None
        else:
            complete_backbone = torchvision.models.__dict__[backbone_name](pretrained=True)

        if backbone_name.startswith("vgg"):
            assert backbone_name in ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
            self.extractor, self.output_features_size = self.vgg(complete_backbone)
        elif backbone_name.startswith("resnet"):
            assert backbone_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
            self.extractor, self.output_features_size = self.resnet(backbone_name, complete_backbone)
        elif backbone_name.startswith("densenet"):
            assert backbone_name in ["densenet121", "densenet161", "densenet169", "densenet201"]
            self.extractor, self.output_features_size = self.densenet(backbone_name, complete_backbone)
        elif backbone_name == "googlenet":
            self.extractor, self.output_features_size = self.googlenet(complete_backbone)
        elif backbone_name == "inception_v3":
            self.extractor, self.output_features_size = self.inception_v3(complete_backbone)
        elif backbone_name == "alexnet":
            self.extractor, self.output_features_size = self.alexnet(complete_backbone)
        elif backbone_name == "mobilenet_v2":
            self.extractor, self.output_features_size = self.mobilenet_v2(complete_backbone)
        elif backbone_name == "mobilenet_v3_small":
            self.extractor, self.output_features_size = self.mobilenet_v3_small(complete_backbone)
        elif backbone_name == "mobilenet_v3_large":
            self.extractor, self.output_features_size = self.mobilenet_v3_large(complete_backbone)
        elif backbone_name == "efficientnet_b0": 
            self.extractor, self.output_features_size = self.efficientnet_b0(complete_backbone)
        elif backbone_name == "efficientnet_b1":
            self.extractor, self.output_features_size = self.efficientnet_b1(complete_backbone)
        elif backbone_name == "efficientnet_b2":
            self.extractor, self.output_features_size = self.efficientnet_b2(complete_backbone)
        elif backbone_name == "efficientnet_b3":
            self.extractor, self.output_features_size = self.efficientnet_b3(complete_backbone)
        elif backbone_name == "efficientnet_b4":
            self.extractor, self.output_features_size = self.efficientnet_b4(complete_backbone)
        elif backbone_name == "efficientnet_b5":
            self.extractor, self.output_features_size = self.efficientnet_b5(complete_backbone)
        elif backbone_name == "efficientnet_b6":
            self.extractor, self.output_features_size = self.efficientnet_b6(complete_backbone)
        elif backbone_name == "efficientnet_b7":
            self.extractor, self.output_features_size = self.efficientnet_b7(complete_backbone)
        elif backbone_name == "efficientnet_v2_s": 
            self.extractor, self.output_features_size = self.efficientnet_v2_s(complete_backbone)
        elif backbone_name == "efficientnet_v2_m":
            self.extractor, self.output_features_size = self.efficientnet_v2_m(complete_backbone)
        elif backbone_name == "efficientnet_v2_l": 
            self.extractor, self.output_features_size = self.efficientnet_v2_l(complete_backbone)
        elif backbone_name == "squeezenet1_0": 
            self.extractor, self.output_features_size = self.squeezenet1_0(complete_backbone)
        elif backbone_name == "squeezenet1_1": 
            self.extractor, self.output_features_size = self.squeezenet1_1(complete_backbone)            
        elif backbone_name == "shufflenet_v2_x0_5":
            self.extractor, self.output_features_size = self.shufflenet_v2_x0_5(complete_backbone)
        elif backbone_name == "shufflenet_v2_x1_0": 
            self.extractor, self.output_features_size = self.shufflenet_v2_x1_0(complete_backbone)  
        elif backbone_name == "shufflenet_v2_x1_5":
            self.extractor, self.output_features_size = self.shufflenet_v2_x1_5(complete_backbone) 
        elif backbone_name == "wide_resnet50_2": 
            self.extractor, self.output_features_size = self.wide_resnet50_2(complete_backbone)
        elif backbone_name == "wide_resnet101_2":
            self.extractor, self.output_features_size = self.wide_resnet101_2(complete_backbone)
        elif backbone_name == "resnext50_32x4d":
            self.extractor, self.output_features_size = self.resnext50_32x4d(complete_backbone)
        elif backbone_name == "resnext101_32x8d":
            self.extractor, self.output_features_size = self.resnext101_32x8d(complete_backbone)
        
        else:
            raise NotImplementedError

    def forward(self, x):
        patch_features = self.extractor(x)

        return patch_features

    def vgg(self, complete_backbone):
        output_features_size = 512 * 7 * 7
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def resnet(self, backbone_name, complete_backbone):
        if backbone_name in ["resnet18", "resnet34"]:
            output_features_size = 512 * 1 * 1
        else:
            output_features_size = 2048 * 1 * 1
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def googlenet(self, complete_backbone):
        output_features_size = 1024 * 1 * 1
        extractor = googlenet(pretrained=True, aux_logits=False)

        return extractor, output_features_size

    def densenet(self, backbone_name, complete_backbone):
        if backbone_name == "densenet121":
            output_features_size = 1024 * 7 * 7
        elif backbone_name == "densenet161":
            output_features_size = 2208 * 7 * 7
        elif backbone_name == "densenet169":
            output_features_size = 1664 * 7 * 7
        else:
            output_features_size = 1920 * 7 * 7
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def inception_v3(self, complete_backbone):
        output_features_size = 2048 * 1 * 1
        extractor = inception_v3(pretrained=True, aux_logits=False)

        return extractor, output_features_size

    def alexnet(self, complete_backbone):
        output_features_size = 256 * 6 * 6
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))
        
        return extractor, output_features_size
    
    def mobilenet_v2(self, complete_backbone):
        output_features_size =  1280 * 7 * 7 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def mobilenet_v3_small(self, complete_backbone):
        output_features_size = 576 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def mobilenet_v3_large(self, complete_backbone):
        output_features_size = 960 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_b0(self, complete_backbone):
        output_features_size = 1280 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
   
    def efficientnet_b1(self, complete_backbone):
        output_features_size = 1280 * 1 * 1
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_b2(self, complete_backbone):
        output_features_size = 1408 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_b3(self, complete_backbone):
        output_features_size = 1536 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_b4(self, complete_backbone):
        output_features_size = 1792 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_b5(self, complete_backbone):
        output_features_size = 2048 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_b6(self, complete_backbone):
        output_features_size = 2304 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_b7(self, complete_backbone):
        output_features_size = 2560 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_v2_s(self, complete_backbone):
        output_features_size = 1280 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_v2_m(self, complete_backbone):
        output_features_size = 1280 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def efficientnet_v2_l(self, complete_backbone):
        output_features_size = 1280 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def squeezenet1_0(self, complete_backbone):
        output_features_size = 1000 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def squeezenet1_1(self, complete_backbone):
        output_features_size = 512 * 13 * 13  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    
    def wide_resnet50_2(self, complete_backbone):
        output_features_size = 2048 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def wide_resnet101_2(self, complete_backbone):
        output_features_size = 2048 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def resnext50_32x4d(self, complete_backbone):
        output_features_size = 2048 * 1 * 1  
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    
    def resnext101_32x8d(self, complete_backbone):
        output_features_size = 2048 * 1 * 1 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def shufflenet_v2_x0_5(self, complete_backbone):
        output_features_size = 1024 * 7 * 7 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
    
    def shufflenet_v2_x1_0(self, complete_backbone):
        output_features_size = 1024 * 7 * 7 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size

    def shufflenet_v2_x1_5(self, complete_backbone):
        output_features_size = 1024 * 7 * 7 
        extractor = nn.Sequential(*(list(complete_backbone.children())[:-1]))

        return extractor, output_features_size
