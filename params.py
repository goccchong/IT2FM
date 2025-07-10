import torch
from thop import profile
from models.SFFNet import SFFNet

#frame_work = FCN8s(num_classes=6)
#frame_work = SegNet(input_nbr=3,label_nbr=6)
#frame_work = PSPNet(num_classes=6, pretrained=True)
#frame_work = UNet(n_channels=3, n_classes=6)
#frame_work = LinkNet(n_classes=6)
#frame_work = CMTFNet()
#frame_work = UNetFormer()
#frame_work = FuzzyNet(n_class=6)

net =   SFFNet(num_classes=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

inputs = torch.randn(1, 3, 512, 512)
inputs = inputs.to("cuda")

flops, params = profile(net, (inputs,))
print('flops: ', flops/ 1e9, 'params: ', params/ 1e6)

