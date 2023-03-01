import paddle 
import paddle.nn as nn
from .swin_video import SwinTransformer3D
from easymia.libs import manager
from easymia.core.abstract_model import AbstractModel

class Backbone(nn.Layer):
    def __init__(self):
        super(Backbone, self).__init__()
        self.swinT3D = SwinTransformer3D(
                patch_size=[2, 4, 4],
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=[8, 7, 7],
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True
            )
        self.swinT3D.init_weights()
        self.pool = nn.AdaptiveAvgPool3D(
            output_size = [1, 1, 1], 
            data_format='NCDHW'
        )

    def forward(self, x):
        x = self.swinT3D(x)
        x = self.pool(x)
        x = paddle.squeeze(x, axis=[2, 3, 4])
        return x

class Class_Head(nn.Layer):
    def __init__(self, num_classes):
        super(Class_Head, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(axis=-1)


    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x  = self.softmax(x)
        return x

@manager.MODELS.add_component
class Swin_video_single(AbstractModel):
    def __init__(self, mode, num_classes=2):
        super(Swin_video_single, self).__init__(mode)
        self.backbone = Backbone()
        self.class_head = Class_Head(num_classes)  
    
    def __clas__(self,  fundus_img, oct_img):
        oct_out = self.backbone(oct_img)
        oct_out = oct_out
        x = self.class_head(oct_out)
        return x
        