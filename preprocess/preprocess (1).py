import numpy as np
import nibabel as nib
import os

def nii_to_numpy(nii_new_files):
    nii = nib.load(nii_new_files)
    data_array = nii.get_fdata()
    affine = nii.affine
    return data_array, affine


def numpy_to_nii(data_array, affine, output_new_files):
    nii = nib.Nifti1Image(data_array, affine)
    nib.save(nii, output_new_files)


def crop_image_and_mask(image_file, mask_file, label_value):
    
    image, image_affine = nii_to_numpy(image_file)
    mask, mask_affine = nii_to_numpy(mask_file)
    
    #print(image.shape, mask.shape)
    
    #unique_values, counts = np.unique(mask, return_counts=True)
    #print("Unique values and their counts in the mask:", dict(zip(unique_values, counts)))
    
    mask = np.round(mask)
    mask[mask >= 1] = 1
    coords = np.argwhere(mask == label_value)
    if coords.size>0:
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        
        z_max = 400+20
        if z_max>np.shape(mask)[2]:
            z_max = np.shape(mask)[2]
        z_min = z_max-96
        
        cropped_image = image[:, :, z_min:z_max]
        cropped_mask = mask[:, :, z_min:z_max]
        
        #crop by z coord with margin
        margin = 5
        #print(z_min, z_max)
        #print(z_min-margin, z_max+1+margin)
        #print(np.shape(image))
        '''if (z_min-margin > 0) and (z_max+1+margin < np.shape(image)[2]):
            cropped_image = image[:, :400, z_min-margin :z_max+1+margin]
            cropped_mask = mask[:, :400, z_min-margin:z_max+1+margin]
        elif (z_max+1+margin < np.shape(image)[2]):
            cropped_image = image[:, :400, z_min : z_max+1+margin*2]
            cropped_mask = mask[:, :400, z_min : z_max+1+margin*2]
        
        else:
            cropped_image = image[:, :400, z_min : z_max+1]
            cropped_mask = mask[:, :400, z_min : z_max+1]'''
        #transform affine if not None
        '''image_affine[:3, 3] += image_affine[:3, :3].dot([x_min, y_min, z_min])
        mask_affine[:3, 3] += mask_affine[:3, :3].dot([x_min, y_min, z_min])'''
        
        return cropped_image, cropped_mask, image_affine, mask_affine
    else:
        return None, None, None, None



def run_crop_and_save(image_new_files, mask_new_files, cropped_image_new_files, cropped_mask_new_files):
    results = crop_image_and_mask(image_new_files, mask_new_files, label_value=1)
    #print(len(results[:2]))
    if all(r is not None for r in results[:2]):
        cropped_image, cropped_mask, image_affine, mask_affine = results
        numpy_to_nii(cropped_image, image_affine, cropped_image_new_files)
        numpy_to_nii(cropped_mask, mask_affine, cropped_mask_new_files)





###새로운 경로에 크롭처리해서 실제로 저장하는 코드 

ori_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\nii_data'
new_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\for_train'
anon_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\anon_data'

patients = os.listdir(ori_dir)
patients = sorted(patients, key=lambda x: int(x[7:]))

for patient in patients:
    
    if patient is not None:
        files = os.listdir(os.path.join(ori_dir, patient))
        
        CT_files = [f for f in files if f.startswith('CT')]
        RT_files = [f for f in files if f.startswith('RT')]
        
        z = len(os.listdir(os.path.join(anon_dir, patient)))
        print(patient, 'started!')
        '''if int(z)<429:
            print(z)
            print(patient)'''
        for RT in RT_files:
            #print(RT)
            label_num = RT.split('-')[1].split('.')[0]
            
            if label_num == '1' :
                new_files = os.path.join(new_dir, 'label'+label_num)
                
                if not os.path.exists(new_files):
                    os.mkdir(new_files)
                if not os.path.exists(os.path.join(new_files, patient)):
                    os.mkdir(os.path.join(new_files, patient))
                    
                image_path = os.path.join(ori_dir, patient, CT_files[0])
                label_path = os.path.join(ori_dir, patient, RT)
                new_image_path = os.path.join(new_files,'data', 'imagesTr', CT_files[0])
                new_mask_path = os.path.join(new_files, 'data', 'labelsTr', RT)
                
                run_crop_and_save(image_path, label_path, new_image_path, new_mask_path)
            





#folder by label - mmkdir code
#confirm all # of dicom data(confirm over 800 -> sorted well or not)
#