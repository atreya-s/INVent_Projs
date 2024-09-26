from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt 



# import libraries
import pandas as pd

import torch
from scipy.ndimage import label, center_of_mass, find_objects


# %%
image_path_base = Path("/Users/atreyasridharan/Desktop/nnUNet_new/imagesTr")
mask_path_base = Path("/Users/atreyasridharan/Desktop/nnUNet_new/labelsTr")

# %%


for image_path in image_path_base.glob("*.mha"):
    mask_path = Path('_'.join(image_path.stem.split("_")[:-1])).with_suffix(".mha")
    mask_path = mask_path_base / mask_path

    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask)

    # print(image.GetSize())
    # print(mask.GetSize())

    plt.imshow(image[54,:,:], cmap='gray')

    
    # plt.contour(mask[54,:,:], levels=[0.5], colors='r')

    unique, counts = np.unique(mask, return_counts=True)

    print(dict(zip(unique, counts)))

    mask1 = mask.copy()
    mask2 = mask.copy()
    mask3 = mask.copy()

    mask1[mask1 == 2] = 0
    mask1[mask1 == 3] = 0


    mask2[mask2 == 1] = 0
    mask2[mask2 == 3] = 0

   

    mask3[mask3 == 1] = 0
    mask3[mask3 == 2] = 0


    plt.contour(mask1[54,:,:], levels=[0.5], colors='b')
    # plt.contour(mask2[54,:,:], levels=[0.5], colors='r')
    # plt.contour(mask3[54,:,:], levels=[0.5], colors='g')


    break

# %%
plt.imshow(image[54,:,:], cmap='gray')
plt.contour(mask3[54,:,:], levels=[0.5], colors='g')
# plt.imshow(mask3[54,:,:], cmap='BrBG', alpha=0.5)



# %%
def add_bbox(mask2d, im2d):
    """
    Add bounding boxes to an image based on a mask.


    Args:
        mask2d (np.array): A 2D binary mask which includes all the areas. mask[0]
        im2d (np.array): A 2D image.

    Returns:
        int: The number of areas.
        list: The bounding boxes of the areas.
    """

    if isinstance(mask2d, torch.Tensor):
        mask2d = mask2d.numpy()


    # negative_mask = np.logical_not(mask2d).astype(int)

    # # Label the connected components in the negative mask
    # labeled_mask, num_features = label(negative_mask, structure=np.ones((3, 3)))


    # mask2d = np.logical_not(mask2d).astype(int)

    # Label the connected components in the negative mask
    labeled_mask, num_features = label(mask2d, structure=np.ones((3, 3)))

    # Find the bounding boxes of the labeled components
    bounding_boxes = find_objects(labeled_mask)

    plt.imshow(im2d, cmap='gray')
    plt.contour(mask2d, levels=[0.5], colors='g')

    for bbox in bounding_boxes:
        y_min, y_max = bbox[0].start, bbox[0].stop
        x_min, x_max = bbox[1].start, bbox[1].stop
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                          edgecolor='red', facecolor='none', linewidth=2))
        
    plt.show()



    return num_features, bounding_boxes



# %%
add_bbox(mask3[54,:,:], image[54,:,:])


# %%
add_bbox(mask1[54,:,:], image[54,:,:])


# %%
add_bbox(mask2[54,:,:], image[54,:,:])
