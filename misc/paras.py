from thop import clever_format
from thop import profile
from models.networks import BASE_Transformer
from models.fresunet import FresUNet
from models.unet import Unet
from models.siamunet_conc import SiamUnet_conc
from models.siamunet_diff import SiamUnet_diff
from models.DSIFN import DSIFN
from models.SNUNet_ECAM import SNUNet_ECAM,Siam_NestedUNet_Conc
from models.DTCDSCN import CDNet_model,SEBasicBlock
from models.ICIFNet import ICIFNet,res_para,pvt_para
from models.CrossNet import CrossNet,CrossNet2,CrossNet3
from models.UNet_mtask import EGRCNN
from models.MixAtt import MixAttNet5,MixAttNet34 #CICNet
from models.MSPSNet import MSPSNet
from models.DARNet import DARNet
# from models.CTFINet import CTFINet
from models.DMINet import DMINet_0916,DMINet50,DMINet,DMINet101
from models.TFIGR import TFIGR
# from models.ReciprocalNet import ReciprocalNet,ReciprocalNet2,ReciprocalNet3,ReciprocalNet4,ReciprocalNet0212,ReciprocalNet0213,ReciprocalNet0214,ReciprocalNet0215,ReciprocalNet0216
# from models.FYCNet import FYCNet0228
from models.TCSVT import TCSVTNet0309,TCSVTNet0310,TCSVTNet0311,TCSVTNet0312,TCSVTNet0313,TCSVTNet0314,TCSVTNet0315,TCSVTNet0316,TCSVTNet0317,TCSVTNet0318,NeurIPS0322,TCSVTNet0316_large,TCSVTNet0316_huge
from models.LCANet_N3C import LCANet_N3C,LCANet
from models.A2Net import BaseNet
from models.p2v import P2VNet,VMCNet
from models.P2V_FYC import ECICNet,EDMINet,P2V_FYC
from models.TDANet import TDANet
import torch

if __name__ == '__main__':
    device = torch.device('cuda:0')
    
    # model = ICIFNet(pretrained=True)
    # res = res_para()
    # pvt = pvt_para()
    input1 = torch.randn(1, 3, 256, 256).cuda()
    input2 = torch.randn(1, 3, 256, 256).cuda()
    model = TDANet().to(device)
    # model = MixAttNet5().to(device)
    # model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=1, dec_depth=8).to(device)
    flops, params = profile(model, inputs=(input1, input2))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)
    
    # out1, out2 = model(input1, input2)
    # print(out1.shape)

    # model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=1, dec_depth=8) # CrossNet(pretrained=True)
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops)
    # print(params)

    # input1 = torch.randn(1, 3, 256, 256).cuda()
    # input2 = torch.randn(1, 3, 256, 256).cuda()
    # model = EGRCNN().to(device)
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # # print("CrossNet2:")
    # print(flops)
    # print(params)
    
    # flops, params = profile(res, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("res:")
    # print(flops)
    # print(params)

    # flops, params = profile(pvt, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("pvt:")
    # print(flops)
    # print(params)

    # model = DSIFN()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("IFNet:")
    # print(flops)
    # print(params)

    # model = SNUNet_ECAM()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("SNUNet:")
    # print(flops)
    # print(params)

    # model = Unet()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("Unet:")
    # print(flops)
    # print(params)  

    # model = SiamUnet_diff()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("SiamUnet_diff:")
    # print(flops)
    # print(params) 

    # model = SiamUnet_conc()
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("SiamUnet_conc:")
    # print(flops)
    # print(params)  


    # model = CDNet_model(3, SEBasicBlock, [3, 4, 6, 3])
    # flops, params = profile(model, inputs=(input1, input2))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("DTCDSCN:")
    # print(flops)
    # print(params)     
                     