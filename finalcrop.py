import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label

def crop_to_tighter_bounding_box_per_slice(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .mha files in the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".mha"):
            # Load the image
            image_path = os.path.join(input_folder, filename)
            image_sitk = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image_sitk)
            
            cropped_slices = []

            # Process each slice individually
            for slice_idx in range(image_array.shape[0]):
                slice_image = image_array[slice_idx]
                
                # Label connected components in the slice
                labeled_slice, num_features = label(slice_image > 0)
                
                largest_area = 0
                largest_bbox = None

                # Find the largest bounding box
                for i in range(1, num_features + 1):
                    # Get the bounding box of each connected component
                    non_zero_indices = np.where(labeled_slice == i)
                    
                    if non_zero_indices[0].size == 0 or non_zero_indices[1].size == 0:
                        continue  # Skip empty components
                    
                    min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
                    min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])

                    # Calculate the area of the bounding box
                    area = (max_y - min_y + 1) * (max_x - min_x + 1)

                    # Update largest bounding box if this one is bigger
                    if area > largest_area:
                        largest_area = area
                        largest_bbox = (min_y, max_y, min_x, max_x)
                
                # Crop to the largest bounding box (tighter)
                if largest_bbox is not None:
                    min_y, max_y, min_x, max_x = largest_bbox
                    
                    # Ensure the crop is tight by strictly using the bounding box limits
                    cropped_slice = slice_image[min_y:max_y+1, min_x:max_x+1]
                    cropped_slices.append(cropped_slice)

            # Convert the cropped slices into a 3D volume
            if cropped_slices:
                cropped_volume = np.stack(cropped_slices, axis=0)
                
                # Convert the numpy array back to SimpleITK Image
                cropped_image_sitk = sitk.GetImageFromArray(cropped_volume)
                cropped_image_sitk.CopyInformation(image_sitk)
                
                # Save the cropped volume
                output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_TighterCropped.mha")
                sitk.WriteImage(cropped_image_sitk, output_path)
                print(f"Saved tightly cropped image to {output_path}")

# Example usage
input_folder = '/mnt/pan/Data7/axs2220/nnUNet_aisc/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Dataset004_BoundingBoxFistula/nnUNet_BBOX2'  # Replace with your input folder path
output_folder = '/mnt/pan/Data7/axs2220/nnUNet_aisc/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Dataset004_BoundingBoxFistula/FinalCrop'
crop_to_tighter_bounding_box_per_slice(input_folder, output_folder)
