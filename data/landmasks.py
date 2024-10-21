from typing import Tuple
from data.pangeo_catalog import get_patch_from_file
from data.coarse import eddy_forcing
# from gz21.data.utils import cyclize_dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import torch
# from paths import LANDMASKS
import os

def expand_for_cnn_spread(land_mask:xr.DataArray,cnn_field_of_view:int,mode:str = "constant"):
    ones_sig = np.ones((cnn_field_of_view,cnn_field_of_view))
    nplandmask = land_mask.values.squeeze()
    if mode == "constant":
        nplandmask = convolve2d(nplandmask,ones_sig,mode = 'valid')
        nplandmask = np.where(nplandmask>0,1,0)
        spread = (cnn_field_of_view - 1)//2
        if spread == 0:
            return land_mask
        land_mask = land_mask.copy().isel(xu_ocean = slice(spread,-spread),yu_ocean = slice(spread,-spread))
        land_mask.data = nplandmask.reshape(land_mask.shape)
        return land_mask
    elif mode == 'wrap':
        nplandmask = convolve2d(nplandmask,ones_sig,mode = 'same',boundary = 'wrap')
        nplandmask = np.where(nplandmask>0,1,0)
        land_mask.data = nplandmask.reshape(land_mask.shape)
        return land_mask
    elif mode == "torch_pool":
        import torch.nn as nn
        pooler=nn.MaxPool2d(cnn_field_of_view,stride=1)
        with torch.no_grad():
            torchlandmask = torch.from_numpy(nplandmask.squeeze()).type(torch.float32)
            torchlandmask = torchlandmask.reshape(1,1,*torchlandmask.shape)
            torchlandmask = pooler(torchlandmask).numpy().squeeze()
        spread = (cnn_field_of_view - 1)//2
        if spread > 0:
            slc = slice(spread,-spread)
            land_mask = land_mask.copy().isel(xu_ocean = slc,yu_ocean = slc)        
        land_mask.data = torchlandmask.reshape(land_mask.shape)
        return land_mask
            



class CoarseGridLandMask:
    def __init__(self,factor:int = 4,cnn_field_of_view:int = 21,torch_flag:bool = True,ylim:Tuple[int,int] = (-85,85)) -> None:
        self.factor =factor
        self.cnn_field_of_view = cnn_field_of_view
        self._interior_land_mask = None
        self._land_mask = None
        hsint = str(abs(hash((factor,cnn_field_of_view,ylim[0],ylim[1]))))
        self.ylim = ylim
        LANDMASKS = '/scratch/cimes/cz3321/MOM6/experiments/double_gyre/postprocess/offline_test/subgrid2/landmasks'
        self._memory_location = os.path.join(LANDMASKS, hsint + '.nc')
        self.mode = "torch_pool"#"constant" #
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_flag =torch_flag
    def generate_masks(self,):
        patch_data, grid_data = get_patch_from_file(1,None,0,'usurf', 'vsurf') 
        for key in patch_data.keys():
            patch_data[key] = xr.where(np.isnan(patch_data[key]), np.random.randn(*patch_data[key].shape),0)
        patch_data = patch_data.load()
        grid_data = grid_data.load()
        forcing = eddy_forcing(patch_data, grid_data, scale=self.factor)
        forcing = np.abs(forcing)
        interior_land_mask = forcing.S_x + forcing.S_y + forcing.usurf + forcing.vsurf
        interior_land_mask = xr.where(interior_land_mask>0,1,0)
        interior_land_mask = interior_land_mask.sel(yu_ocean = slice(self.ylim[0],self.ylim[1]))
        _interior_land_mask = 1 - expand_for_cnn_spread(interior_land_mask,self.cnn_field_of_view,mode = self.mode)
        
        
        
        patch_data, _ = get_patch_from_file(1,None,0,'usurf', 'vsurf') 
        for key in patch_data.keys():
            patch_data[key] = xr.where(np.isnan(patch_data[key]), 1,0)
        usurf = patch_data.usurf
        usurf = usurf.coarsen({'xu_ocean': int(self.factor),'yu_ocean': int(self.factor)},boundary='trim')
        
        land_density = usurf.mean()
        land_mask = xr.where(land_density >= 0.5,1,0)
        land_mask = land_mask.sel(yu_ocean = slice(self.ylim[0],self.ylim[1]))
        _land_mask= 1 - expand_for_cnn_spread(land_mask,self.cnn_field_of_view,mode = self.mode)
        _interior_land_mask.name = 'interior'
        _land_mask.name = 'default'
        masks = xr.merge([_interior_land_mask,_land_mask])
        masks = masks.isel(time = 0)
        return masks
    def save_to_file(self,):
        masks = self.generate_masks()
        masks.to_netcdf(self._memory_location)
        print(f'{self._memory_location} saved')
    def read_from_file(self,):
        return xr.open_dataset(self._memory_location)
    @property
    def interior_land_mask(self,):
        if self._interior_land_mask is None:
            masks = self.read_from_file()
            self._interior_land_mask = masks.interior
            if self.torch_flag:
                vals = self._interior_land_mask.values.squeeze()
                vals = np.stack([vals],axis = 0)                
                self._interior_land_mask = torch.from_numpy(vals).to(dtype = torch.float32)
        return self._interior_land_mask
    @property
    def land_mask(self,):
        if self._land_mask is None:
            masks = self.read_from_file()
            self._land_mask = masks.default
            if self.torch_flag:
                vals = self._land_mask.values.squeeze()
                vals = np.stack([vals],axis = 0)
                self._land_mask = torch.from_numpy(vals).to(dtype = torch.float32)
        return self._land_mask



def main():
    # cglm = CoarseGridLandMask(torch_flag=False,cnn_field_of_view=21,mode = "constant")
    # cglm.save_to_file()
    # intmap0 = cglm.interior_land_mask.copy()
    # cglm = CoarseGridLandMask(torch_flag=False,cnn_field_of_view=21,mode = "wrap")
    # cglm.save_to_file()
    # intmap1 = cglm.interior_land_mask.copy()
    # cglm = CoarseGridLandMask(torch_flag=False,cnn_field_of_view=21,mode = "torch_pool")
    # cglm.save_to_file()
    # intmap2 = cglm.interior_land_mask.copy()
    
    # diff = intmap1 - intmap0
    # print(f"err10 = {np.sum(np.abs(diff.values)).item()}")
    # diff = intmap2 - intmap0
    # print(f"err20 = {np.sum(np.abs(diff.values)).item()}")
    # # diff.plot()
    # # plt.savefig('difference.png')
    # # plt.close()
    
    # return
    cglm = CoarseGridLandMask(torch_flag=False,cnn_field_of_view=1,)
    cglm.save_to_file()
    cglm.interior_land_mask.plot()
    plt.savefig('coarse_interior_land_mask.png')
    plt.close()
    
    cglm.land_mask.plot()
    plt.savefig('coarse_land_mask.png')
    plt.close()
    
    cglm = CoarseGridLandMask(torch_flag=False,cnn_field_of_view=21,)
    cglm.save_to_file()
    
    cglm.interior_land_mask.plot()
    plt.savefig('coarse_interior_land_mask_expanded.png')
    plt.close()

    cglm.land_mask.plot()
    plt.savefig('coarse_land_mask_expanded.png')
    plt.close()
    
if __name__ == '__main__':
    main()