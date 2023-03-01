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
            output_size = [None, 1, 1], 
            data_format='NCDHW'
        )

    def forward(self, x):
        x = self.swinT3D(x)
        x = self.pool(x)
        x = paddle.squeeze(x, axis=[3, 4])
        return x

class Fundus_Backbone(nn.Layer):
    def __init__(self,
                padding_type='SAME',
               override_params=None,
               use_se=True,
               pretrained=None,
               use_ssld=False,):
        super(Fundus_Backbone, self).__init__()
        self.fundus_branch = paddle.vision.models.resnet101(pretrained=True)
        self.linear = nn.Linear(1000, 16)

    def forward(self, x):
        x = self.fundus_branch(x)
        x = self.linear(x)
        return x

class Connection(nn.Layer):
    def __init__(self):
        super(Connection, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(16, nhead=2, dim_feedforward=200, dropout=0.2,
                                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(-0.1, 0.1)),
                                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0)))
        self.sequence_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1D(output_size = 1)

    def forward(self, x):
        x = self.sequence_encoder(x)
        x = self.pool(x)
        x = x.squeeze(axis = 2)
        return x


class Class_Head1(nn.Layer):
    def __init__(self, num_classes):
        super(Class_Head1, self).__init__()
        self.linear1 = nn.Linear(1040, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(axis=-1)
        self.pool = nn.AdaptiveAvgPool1D(
            output_size = 1
        )

    def forward(self, fundus_out, oct_out):
        oct_out = self.pool(oct_out)
        oct_out = oct_out.squeeze(axis = 2)
        x = paddle.concat(x = [oct_out, fundus_out], axis=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x  = self.softmax(x)
        return x

class Class_Head2(nn.Layer):
    def __init__(self, num_classes):
        super(Class_Head2, self).__init__()
        self.pool = nn.AdaptiveAvgPool1D(output_size = 1)
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
class Swin_video_1(AbstractModel):
    def __init__(self, mode, num_classes=2):
        super(Swin_video_1, self).__init__(mode)
        self.fundus_backbone = Fundus_Backbone()
        self.backbone = Backbone()
        self.neck = Connection()
        self.class_head1 = Class_Head1(num_classes)
        self.class_head2 = Class_Head2(num_classes)  
        self.position_embedding = paddle.create_parameter(
                shape=[1, 16],
                dtype='float32',
                default_initializer=nn.initializer.TruncatedNormal(std=.02)) 
    
    def __clas__(self,  fundus_img, oct_img):
        fundus_out = self.fundus_backbone(fundus_img)
        oct_out = self.backbone(oct_img)
        stage_out1 = self.class_head1(fundus_out, oct_out)
        oct_out = oct_out+self.position_embedding
        oct_out = oct_out.transpose([1, 0, 2])
        x = paddle.multiply(oct_out, fundus_out)
        x = x.transpose([1, 0, 2])
        x = self.neck(x)
        stage_out2 = self.class_head2(x)
        if self.training:
            return stage_out2, stage_out1
        else:
            return stage_out2
        