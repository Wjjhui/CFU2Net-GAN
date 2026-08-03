# from matplotlib import pyplot as plt
import torch
import einops
import numpy
from matplotlib import pyplot
def rainprint(x:torch.tensor,img_path="result/sample.jpg",remove_background=False,dpi=300,vmax=10,vmin = 0,renormalization=False):
    # x = x.detach().cpu().numpy()
    # print(x.shape)
    # x = x.detach().cpu()
    # out = einops.rearrange(x.numpy(), "b t w h -> (b w) (t h)")
    # out = numpy.clip(out, 0, 10.0)

    out = einops.rearrange(torch.squeeze(x.detach().cpu()).numpy(),"b t w h -> (b w) (t h)")

    if renormalization:
        out = out*22.0
    pyplot.figure(figsize=(9, 9))
    pyplot.axes()
    pyplot.axis('off')
    pyplot.imshow(out, vmax=10, vmin=0,cmap='jet')
    
    if remove_background:
        out[out<0.01] = numpy.nan
    pyplot.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    
    pyplot.close()
    
    

    
    
    
    