import os
import torch


import numpy as np
import pydicom as dcm
import os
import matplotlib.pyplot as plt
from skimage.draw import polygon
import nibabel as nib
import math
import cv2
import pandas as pd
import scipy.ndimage


ori_dir = 'D:\\liver-new\\anonymize\\new Arterial data_2021.01'
patients = os.listdir(ori_dir)
save_dir = 'D:\\sevdata_preprocess\\result-debug'

n_x = 512
pix_spacing = []
pix_spacing_z = []

#pix_smallest = 0.54299998283386 (22) pix_biggest = 1.00699996948242 (378)
pix_middle = 1.0


patients = sorted(patients, key=lambda x: int(x.split("\\")[0]))
#print('mmmmmmm')
#print(patients[0].split("\\"))
print(patients)
for patient_idx, patient in enumerate(patients):
    if patient_idx>=0:

        #------ct-------


        #patient_dir = os.path.join(save_dir, str(patient_idx+1))
        #if not os.path.exists(patient_dir):
        #    os.makedirs(patient_dir)

        data_dir = os.path.join(ori_dir, patient, 'CT')

        lst_data = os.listdir(data_dir)
        #print(data_dir)
        
        lst_image = [f for f in lst_data if f.endswith('.dcm')]
        image = np.zeros((512,512,len(lst_image)), dtype=np.uint16)

        patient_pos = []
        CT_x_pos = np.zeros(512)
        CT_y_pos = np.zeros(512)
        CT_z_pos = np.zeros(len(lst_image))
        img_ori_z = []
        for i in range(len(lst_image)):
            raw_file = dcm.dcmread(os.path.join(data_dir, lst_image[i]))
            raw_file = dcm.read_file(os.path.join(data_dir, lst_image[i]))
            img_ori_z.append(raw_file.ImagePositionPatient[2])
            file = (raw_file.pixel_array)
            #print(type(file[1,1]))
            image[:,:,i]=file
        #print(file.shape)
        sort_z = sorted(img_ori_z, reverse=True)
        for i in range(512):
            CT_x_pos[i] = raw_file.ImagePositionPatient[0] + raw_file.PixelSpacing[0]*(i)
            CT_y_pos[i] = raw_file.ImagePositionPatient[1] + raw_file.PixelSpacing[1]*(i)

        for i in range(len(lst_image)):
            CT_z_pos[i] = img_ori_z[0] + raw_file.SliceThickness*(i)

        #---------------중요 " image_sub에 image sort_z와 순서 맞춰서 다시저장해야지 CT이미지와 RT이미지의 순서를 맞출 수 있음!!!
        image_sub = np.zeros((n_x, n_x, len(lst_image)), dtype=np.uint16)
        for i in range(len(lst_image)):
            for i_1 in range(len(lst_image)):
                if sort_z[i]==img_ori_z[i_1]:
                    image_sub[:,:,i] = image[:,:,i_1]

        #------------rt-------------


        data_dir1= os.path.join(ori_dir, patient, 'RTst')
        lst_data1 = os.listdir(data_dir1)
        lst_image1 = [f for f in lst_data1 if f.endswith('.dcm')]
        raw_file_RT = dcm.dcmread(os.path.join(data_dir1, lst_image1[0]))
        structures = {}
        for item in raw_file_RT.StructureSetROISequence:
            structures[item.ROINumber] = item.ROIName
        ROI_seq = raw_file_RT.ROIContourSequence
        # print(raw_file_RT)
        contours=[]
        # print(len(ROI_seq))

        
        imageRT = np.zeros((n_x, n_x, len(lst_image)), dtype=np.uint8)

        for i in range(len(ROI_seq)):
            # print(len(ROI_seq[i].ContourSequence))
            contour={}
            contour['color'] = tuple(np.array(ROI_seq[i].ROIDisplayColor)/255.0)
            
            #contour['number'] = 0
            contour['name'] = raw_file_RT.StructureSetROISequence[i].ROIName
            #assert contour['number'] == raw_file_RT.StructureSetROISequence[i].ROINumber
            contour['contours'] = [s.ContourData for s in ROI_seq[i].ContourSequence]
            contours.append(contour)
        



        #name에 대해 고유 식별번호 부여 -> 해당 name이면 해당 번호를 불러오면 됨
        color_name = {}

        result_num = 3
        #tumor:red, liver:purple, roi-1:green, others:blue
        for contour in contours:
            if contour['name'] == 'liver':
                contour['number'] = 1
                contour['color'] = (1,0,1)
                color_name[contour['color']] = 'purple'
            elif contour['name'] == 'Liver':
                contour['number'] = 1
                contour['color'] = (1,0,1)
                color_name[contour['color']] = 'purple'
        
            elif contour['name'] == 'tumor':
                contour['number'] = 2
                contour['color'] = (1,0,0)
                color_name[contour['color']] = 'red'
            elif contour['name'] == 'Tumor':
                contour['number'] = 2
                contour['color'] = (1,0,0)
                color_name[contour['color']] = 'red'

            elif contour['name'] == 'TUmor':
                contour['number'] = 2
                contour['color'] = (1,0,0)
                color_name[contour['color']] = 'red'
            
            elif contour['name'] == 'TUMOR':
                contour['number'] = 2
                contour['color'] = (1,0,0)
                color_name[contour['color']] = 'red'

            else:
                pass
            '''elif contour['name'] == 'ROI-1':
                contour['number'] = 3
                contour['color'] = (0,1,0)
                color_name[contour['color']] = 'green'
            else:
                contour['number']=0
                contour['color'] = (0,0,1)
                color_name[contour['color']] = 'blue' '''

        
        
        #print(type(image_sub))



        image_sub = np.rot90(image_sub, k=1, axes=(0,1))
        image_sub = np.flip(image_sub, axis=0)

        image_sub1 = np.zeros((512,512,len(lst_image)), dtype=np.uint16)

        ratio = raw_file.PixelSpacing[0]/pix_middle
        '''def resize_slice(image_sub, ratio):

            for i in range(len(lst_image)):
                # 각 2D 슬라이스에 대해 리사이징 수행
                image_temp = scipy.ndimage.zoom(image_sub[:, :, i], ratio, order=3)

                # 리사이징된 이미지를 중앙에 배치
                start_row = (image_sub1.shape[0] - image_temp.shape[0]) // 2
                start_col = (image_sub1.shape[1] - image_temp.shape[1]) // 2
                image_sub1[start_row:start_row + image_temp.shape[0], start_col:start_col + image_temp.shape[1], i] = image_temp

            return image_sub1

        image_sub = resize_slice(image_sub, ratio)
        print(image_sub.shape)'''

        '''if raw_file.PixelSpacing[0] != pix_smallest:
            image_sub1 = np.zeros((n_x, n_x, len(lst_image)), dtype=np.int16)
            for i in range(len(lst_image)):
                temp = image_sub[:,:,i]
                image_temp = cv2.resize(temp,dsize=(round(n_x*ratio),round(n_x*ratio)),
                                        interpolation=cv2.INTER_LINEAR)
                
                
                start_row = (image_sub1.shape[0] - image_temp.shape[0]) // 2
                start_col = (image_sub1.shape[1] - image_temp.shape[1]) // 2
                image_sub1[start_row:start_row + image_temp.shape[0], start_col:start_col + image_temp.shape[1], i] = image_temp
            image_sub = image_sub1
        else:
            pass'''
        if raw_file.PixelSpacing[0] != pix_middle:
            image_sub1 = np.zeros((n_x, n_x, len(lst_image)), dtype=np.uint16)
            for i in range(len(lst_image)):
                temp = image_sub[:,:,i]
                image_temp = cv2.resize(temp,dsize=(round(n_x*raw_file.PixelSpacing[0]/pix_middle),round(n_x*raw_file.PixelSpacing[0]/pix_middle)),interpolation=cv2.INTER_LINEAR)
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
        #print(np.max(image_sub))
        image_sub = image_sub -1024.0 #.0안해서 uint 연산 안됐어서 수정함
        #array를 space_

        img1 = nib.Nifti1Image(image_sub, None, dtype=np.uint16)
        #512*512똑같은 크기인데, pixel spacing이 다르면 resolution이 달라짐 -> 더 큰 해상도를 가지고있는 애들을, 고정된 크기 스페이싱으로 줄인다음(array shape에대해 비율만큼 곱해주기)에
        #np.zeros로 중앙에 박아넣는작업 필요
        pix_spacing.append([raw_file.PixelSpacing[0], patient])
        pix_spacing_z.append([sort_z[11]-sort_z[12], patient])
        #if patient_idx%50 == 0:
        #    print(pix_spacing)
        
        img1.header.get_xyzt_units()

        img1.header['pixdim'][0] = 1
        img1.header['pixdim'][1] = pix_middle
        img1.header['pixdim'][2] = pix_middle
        img1.header['pixdim'][3] = sort_z[10]-sort_z[11]
        
        
        #제거항목
        #img.header['pixdim'][3] = raw_file.SliceThickness
        img1.to_filename(os.path.join(save_dir, 'CT' + str(patient_idx + 1) + '.nii.gz'))
        #print(img1.header)
    #CT_z_pos였음 원래
        
        
        #mask_s = np.zeros((n_x, n_x), dtype=np.uint8)


        for i in range(len(sort_z)):

            z = sort_z[i]
            #CT이미지 먼저 올리기 -> RT이미지 그위에 그리기
            #plt.imshow(image_sub[:, :, i],cmap='gray')
            
            
            #CT_dir = os.path.join(patient_dir, 'CT')
            #RT_dir = os.path.join(patient_dir, 'RT')
            #if not os.path.exists(CT_dir):
            #    os.makedirs(CT_dir)
            #if not os.path.exists(RT_dir):
            #    os.makedirs(RT_dir)
            
            #save_dir_sub_CT = os.path.join(CT_dir, str(i+1))
            #plt.savefig(save_dir_sub_CT)
            #plt.clf()
            



            #mask = np.zeros((n_x, n_x), dtype=np.uint8)
            for con in contours:
                #plt.imshow(image_sub[:, :, i],cmap='gray')

                
                for i_1 in range(result_num+1):  # i_1 = 0,1,2,3의 number
                    if 'number' in con:
                        if con['number'] == i_1:
                            #print(con['number'])
                            for k in range(len(con['contours'])):
                                if con['contours'][k] is not None:
                                    corr = []
                                    j = 0
                                    p1 = []
                                    p2 = []
                                    p3 = []
                                    
                                    #print(con['contours'])
                                    #print('mmmmmmmmmm',str(patient_idx+1), 'is done!', data_dir)
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
                                            # plt.plot((p1[j]-CT_x_pos[0])/raw_file.PixelSpacing[0],(p2[j]-CT_y_pos[0])/raw_file.PixelSpacing[1],'ro')
                                    
                                    
                                    if len(corr)>0:
                                        #xs, ys = zip(*corr)
                                        #plt.plot(xs,ys,color=con['color'])
                                        
                                        polygon1 = np.array(corr, dtype=np.int32)
                                        mask = np.zeros((n_x, n_x), dtype=np.uint8)
                                                
                                        #cv2.polylines(mask, [polygon1], True, con['color'], 1)
                                        cv2.fillPoly(mask, [polygon1], con['color'])
                                        #print(polygon1)
                                        #m = pd.DataFrame(mask)
                                        #m.value_counts()
                                        #print(np.unique(mask*i_1))
                                        
                                        #for i_2 in range(n_x):
                                        #    for i_3 in range(n_x):
                                        #        if imageRT[i_2,i_3,i] !=0 and mask[i_2,i_3]!=0:
                                        #            mask_s[i_2,i_3] = 0
                                        #        else:
                                        #            mask_s[i_2,i_3] = mask[i_2,i_3]
                                        #print(corr)
                                        #print(np.shape(corr), np.shape(mask), np.shape(mask_s))
                                        
                                        mask_indices = np.where(mask != 0)
                                        
                                        #print(np.unique(mask*i_1))
                                        
                                        #yys, xxs = mask_index

                                        #plt.scatter(xxs, yys, c=color_name[con['color']])

                                        #imageRT[:, :, i] += mask
                                        imageRT[mask_indices[0], mask_indices[1], i] += mask[mask_indices[0], mask_indices[1]]
                    else:
                        pass                
        #print('mmm',np.unique(imageRT))
                

                #RT_sub_dir = os.path.join(RT_dir ,str(con['number']))
        #if not os.path.exists(RT_sub_dir):
        #            os.makedirs(RT_sub_dir)

        #        save_dir_sub_RT = os.path.join(RT_sub_dir, str(i+1))
        #        
        #        plt.savefig(save_dir_sub_RT)
        #        plt.clf()
        #print(np.unique(imageRT))
                    
        
        
        imageRT = np.rot90(imageRT, k=1, axes=(0,1))
        imageRT = np.flip(imageRT, axis=0)
        
        if raw_file.PixelSpacing[0] != pix_middle:
            imageRT1 = np.zeros((n_x, n_x, len(lst_image)), dtype=np.int8)
            for i in range(len(lst_image)):
                RTtemp = imageRT[:,:,i]
                imageRT_temp = cv2.resize(RTtemp,dsize=(round(n_x*raw_file.PixelSpacing[0]/pix_middle),round(n_x*raw_file.PixelSpacing[0]/pix_middle)),
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
        img2.to_filename(os.path.join(save_dir, 'RTst' + str(patient_idx + 1) + '.nii.gz'))
        print('mmmmmmmmmm',str(patient_idx+1), 'is done!')
        


pix_np = np.array(pix_spacing)
pix_z_np = np.array(pix_spacing_z)
#print('pix:--------------------------\n',pix_np)
#print('pix_z:------------------------\n', pix_z_np)
#print(np.min(pix_np), np.max(pix_np))

#print(np.min(pix_z_np), np.max(pix_z_np))
pix_val = pix_np[:,0].astype(np.float64)
pix_z_val = pix_z_np[:,0].astype(np.float64)
max_index = np.unravel_index(np.argmax(pix_val), pix_val.shape)
min_index = np.unravel_index(np.argmin(pix_val), pix_val.shape)
max_index_z = np.unravel_index(np.argmax(pix_z_val), pix_z_val.shape)
min_index_z = np.unravel_index(np.argmin(pix_z_val), pix_z_val.shape)
print('max: ',pix_np[max_index], 'min:',pix_np[min_index])
print('max: ',pix_z_np[max_index_z], 'min: ',pix_z_np[min_index_z])

'''        pix_spacing.append([raw_file.PixelSpacing[0], data_dir])
        #print(raw_file.PixelSpacing[1])
        pix_spacing_z.append([sort_z[10]-sort_z[11], data_dir])
print(pix_spacing)
print(pix_spacing_z)
min_pix = min(pix_spacing, key=lambda x: x[0])
max_pix = max(pix_spacing, key=lambda x: x[0])
print(min_pix, max_pix)
min_z = min(pix_spacing_z, key=lambda x: x[0])
max_z = max(pix_spacing_z, key=lambda x: x[0])
print(min_z, max_z)'''
