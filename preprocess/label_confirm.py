import os
import pandas as pd
import numpy as np
import pydicom as dcm
import cv2
import nibabel as nib
import math

#ori_dir = "C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\data\\all_st"
source_dir = "C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\new_data"
save_dir = "C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\nii_data"
import os

# 지정된 경로 설정 (예: 'path/to/directory')
base_path = "C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\for_train"

# patient1부터 patient417까지의 폴더 생성
for i in range(1, 418):
    folder_name = f"patient{i}"
    folder_path = os.path.join(base_path, folder_name)
    
    # 해당 경로에 폴더가 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
###--------------data anonymize---------------
"""
patient_lst = os.listdir(source_dir)
patient_lst = sorted(patient_lst, key=lambda x: int(x[7:]))
#print(patient_lst)
no_contour = [] # patient 4 only
ROI_len = []
compare_name = []
ref_contours = []

def adjust_labels(patient_lst, start_num, end_num):
    
    for patient in patient_lst:
        if patient == 'patient5':
            files = os.listdir(os.path.join(source_dir, patient))
            data_dir = os.path.join(source_dir, patient)
            RT_files = [f for f in files if f.startswith('RT')]
    
            raw_RT = dcm.dcmread(os.path.join(data_dir, RT_files[0]))
            ROI_seq = raw_RT.ROIContourSequence
            contours = []
            for i in range(len(ROI_seq)):
                contour = {}
                compare_name.append(raw_RT.StructureSetROISequence[i].ROIName)
            for i in range(len(compare_name)):
                contour = {}
                contour['name'] = compare_name[i]
                contour['number'] = i+1
                ref_contours.append(contour)
    compare_name.append('Brain')
    ref_contours.append({'name': 'Brain', 'number': 26})
    print(compare_name)
    print(ref_contours)
    
#patient5 standard -> same label number on same name    
    for patient in patient_lst:
        idx = int(patient[7:])
        if start_num <= idx < end_num+1:
            files = os.listdir(os.path.join(source_dir, patient))
            data_dir = os.path.join(source_dir, patient)
            CT_files = [f for f in files if f.startswith('CT')]
            RT_files = [f for f in files if f.startswith('RT')]
            
            raw_RT = dcm.dcmread(os.path.join(data_dir, RT_files[0]))
            '''structures = {}
            for item in raw_RT.StructureSetROISequence:
                structures[item.ROINumber] = item.ROIName'''
            
            ROI_seq = raw_RT.ROIContourSequence
            
            print(f"==========patient {idx} ============")
            
            contours = []
            name = []
            for i in range(len(ROI_seq)):
                contour = {}
                contour['name'] = raw_RT.StructureSetROISequence[i].ROIName
                contour['contours'] = [s.ContourData for s in ROI_seq[i].ContourSequence]
                contour['color'] = (1,0,0)
                for ref in ref_contours:
                    if ref['name'] == contour['name']:
                        contour['number'] = ref['number']
                contours.append(contour)
                
                #print(contour['number'], contour['name'])
                name.append(raw_RT.StructureSetROISequence[i].ROIName)
            '''if not set(name).intersection(set(compare_name)):
                print(patient)'''
            '''if set(name).issubset(set(compare_name)):
                print("모든 원소가 compare_name 리스트 내에 존재합니다.")
            else:
                print(patient, "---------------------------------------에서 name의 원소가 compare_name 리스트에 존재하지 않습니다.")
                #print(len(ROI_seq), name)
                #print(compare_name)
                # name 리스트의 원소 중 compare_name 리스트에 속하지 않는 원소를 찾아 프린트
                non_matching_elements = set(name) - set(compare_name)
                if non_matching_elements:
                    print(f"{patient}에서 속하지 않는 원소: {', '.join(non_matching_elements)}")
                else:
                    print(f"{patient}의 모든 name 원소가 compare_name 리스트에 속합니다.")'''

            
            imageRT = np.zeros((512, 512, len(CT_files)), dtype=np.uint8)
            image = np.zeros((512,512,len(CT_files)), dtype=np.uint16)
            patient_pos = []
            CT_x_pos = np.zeros(512)
            CT_y_pos = np.zeros(512)
            CT_z_pos = np.zeros(len(CT_files))
            img_ori_z = []
            for i in range(len(CT_files)):
                raw_file = dcm.dcmread(os.path.join(data_dir, CT_files[i]))
                raw_file = dcm.read_file(os.path.join(data_dir, CT_files[i]))
                img_ori_z.append(raw_file.ImagePositionPatient[2])
                file = (raw_file.pixel_array)
                image[:,:,i]=file
            
            sort_z = sorted(img_ori_z, reverse=True)
            for i in range(512):
                CT_x_pos[i] = raw_file.ImagePositionPatient[0] + raw_file.PixelSpacing[0]*(i)
                CT_y_pos[i] = raw_file.ImagePositionPatient[1] + raw_file.PixelSpacing[1]*(i)
            for i in range(len(CT_files)):
                CT_z_pos[i] = img_ori_z[0] + raw_file.SliceThickness*(i)

            image_sub = np.zeros((512, 512, len(CT_files)), dtype=np.uint16)
            for i in range(len(CT_files)):
                for i_1 in range(len(CT_files)):
                    if sort_z[i]==img_ori_z[i_1]:
                        image_sub[:,:,i] = image[:,:,i_1]
            image_sub = np.rot90(image_sub, k=1, axes=(0,1))
            image_sub = np.flip(image_sub, axis=0)
            image_sub1 = np.zeros((512,512,len(CT_files)), dtype=np.uint16)
            pix_middle = 1.0
            ratio = raw_file.PixelSpacing[0]/pix_middle

            if raw_file.PixelSpacing[0] != pix_middle:
                image_sub1 = np.zeros((512, 512, len(CT_files)), dtype=np.uint16)
                for i in range(len(CT_files)):
                    temp = image_sub[:,:,i]
                    image_temp = cv2.resize(temp,dsize=(round(512*raw_file.PixelSpacing[0]/pix_middle),round(512*raw_file.PixelSpacing[0]/pix_middle)),interpolation=cv2.INTER_LINEAR)
                    #print(image_temp.shape)
                    if image_temp.shape[0]>=512:
                        image_sub1[:,:,i] = image_temp[round(image_temp.shape[0]/2)-256:round(image_temp.shape[0]/2)+256,round(image_temp.shape[0]/2)-256:round(image_temp.shape[0]/2)+256]

                    else:
                        if (image_temp.shape[0]%2)==1:
                            #image_sub1[256-round(image_temp.shape[0]/2):256+round(image_temp.shape[0]/2)-1,256-round(image_temp.shape[0]/2):256+round(image_temp.shape[0]/2)-1,i] = image_temp
                            image_sub1[(512-image_temp.shape[0])//2:512-((512-image_temp.shape[0])//2+1),(512-image_temp.shape[0])//2:512-((512-image_temp.shape[0])//2+1),i] = image_temp

                        else:
                            image_sub1[256 - round(image_temp.shape[0] / 2):256 + round(image_temp.shape[0] / 2),
                            256 - round(image_temp.shape[0] / 2):256 + round(image_temp.shape[0] / 2), i] = image_temp
                image_sub = image_sub1
            #print('CT: ',image_sub.shape)
            image_sub = image_sub -1024.0
            img1 = nib.Nifti1Image(image_sub, None, dtype=np.uint16)
            img1.header.get_xyzt_units()

            img1.header['pixdim'][0] = 1
            img1.header['pixdim'][1] = pix_middle
            img1.header['pixdim'][2] = pix_middle
            img1.header['pixdim'][3] = sort_z[10]-sort_z[11]
            img1.to_filename(os.path.join(save_dir, patient, 'CT' + str(idx) + '.nii.gz'))
            
            for con in contours:
                imageRT = np.zeros((512, 512, len(CT_files)), dtype=np.uint8)
                
                if 'number' in con:
                    
                    for i in range(len(sort_z)):

                        z = sort_z[i]
                        
                    
                        
                        for k in range(len(con['contours'])):
                            if con['contours'][k] is not None:
                                corr = []
                                j = 0
                                p1 = []
                                p2 = []
                                p3 = []
                                
                                while j <len(con['contours'][k])/3:
                                    p1.append(con['contours'][k][0 + 3 * j])
                                    p2.append(con['contours'][k][1 + 3 * j])
                                    p3.append(con['contours'][k][2 + 3 * j])
                                    j=j+1
                                    
                                z_coord = np.unique(p3)
                                z_mid = z_coord[math.floor(len(z_coord) / 2)]

                                for j in range(len(p3)):
                                
                                    if abs(z_mid - z) < ((sort_z[0]-sort_z[1]) / 2):
                                    #if p3[j]==int(z):
                                        
                                        corr.append([(p1[j]-CT_x_pos[0])/raw_file.PixelSpacing[0],(p2[j]-CT_y_pos[0])/raw_file.PixelSpacing[1]])
                                        
                                if len(corr)>0:
                                    
                                    polygon1 = np.array(corr, dtype=np.int32)
                                    mask = np.zeros((512, 512), dtype=np.uint8)
                                            
                                    #cv2.polylines(mask, [polygon1], True, con['color'], 1)
                                    cv2.fillPoly(mask, [polygon1], con['color'])
                                    
                                    mask_indices = np.where(mask != 0)
                                    imageRT[mask_indices[0], mask_indices[1], i] += mask[mask_indices[0], mask_indices[1]]
                    imageRT = np.rot90(imageRT, k=1, axes=(0,1))
                    imageRT = np.flip(imageRT, axis=0)
                    if raw_file.PixelSpacing[0] != pix_middle:
                        imageRT1 = np.zeros((512, 512, len(CT_files)), dtype=np.int8)
                        for i in range(len(CT_files)):
                            RTtemp = imageRT[:,:,i]
                            imageRT_temp = cv2.resize(RTtemp,dsize=(round(512*raw_file.PixelSpacing[0]/pix_middle),round(512*raw_file.PixelSpacing[0]/pix_middle)),
                                                    interpolation=cv2.INTER_NEAREST)
                            
                            
                            if imageRT_temp.shape[0]>=512:
                                imageRT1[:,:,i] = imageRT_temp[round(imageRT_temp.shape[0]/2)-256:round(imageRT_temp.shape[0]/2)+256,round(imageRT_temp.shape[0]/2)-256:round(imageRT_temp.shape[0]/2)+256]

                            else:
                                if (imageRT_temp.shape[0]%2)==1:
                                    #imageRT1[256-round(imageRT_temp.shape[0]/2):256+round(imageRT_temp.shape[0]/2)-1,256-round(imageRT_temp.shape[0]/2):256+round(imageRT_temp.shape[0]/2)-1,i] = imageRT_temp
                                    imageRT1[(512-imageRT_temp.shape[0])//2:512-((512-imageRT_temp.shape[0])//2+1),(512-imageRT_temp.shape[0])//2:512-((512-imageRT_temp.shape[0])//2+1),i] = imageRT_temp
                                else:
                                    imageRT1[256 - round(imageRT_temp.shape[0] / 2):256 + round(imageRT_temp.shape[0] / 2),
                                    256 - round(imageRT_temp.shape[0] / 2):256 + round(imageRT_temp.shape[0] / 2), i] = imageRT_temp
                        imageRT = imageRT1
                        #print('RT: ', imageRT.shape)
                    else:
                        pass
                    img2 = nib.Nifti1Image(imageRT, None, dtype=np.uint16)
                    img2.header.get_xyzt_units()

                    img2.header['pixdim'][0] = 1
                    img2.header['pixdim'][1] = pix_middle
                    img2.header['pixdim'][2] = pix_middle
                    img2.header['pixdim'][3] = sort_z[10]-sort_z[11]
                    
                    label = img2.get_fdata()
                    label[label>=3] = 0
                    
                    img2 = nib.Nifti1Image(label, img2.affine, img2.header)
                    
                    print(np.max(img2.get_fdata()))
                    img2.to_filename(os.path.join(save_dir, patient, 'RTst' + str(idx) + '-' + str(con['number']) + '.nii.gz'))
                else:
                    pass
                            
                                
                                
adjust_labels(patient_lst, 1, 417)"""