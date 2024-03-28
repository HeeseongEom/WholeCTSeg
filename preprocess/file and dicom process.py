import os
import pandas as pd
import numpy as np
import pydicom as dcm



#new_dir = "C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\data\\all_st"
#new_dir = "C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\anon_data"
#new_dir = "C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\anon"


'''def rename_subfolders(dir):
    
    # 상위 폴더 내의 모든 항목을 가져오고 폴더만 필터링
    subfolders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    # 폴더들을 순회하며 이름 변경
    for idx, folder in enumerate(subfolders, start=1):
        new_dirl_path = os.path.join(dir, folder)
        new_path = os.path.join(dir, f'patient{idx}')
        os.rename(new_dirl_path, new_path)
        print(f"Renamed: {folder} to {'patient' + str(idx)}")

rename_subfolders(new_dir)

'''
###--------------data anonymize---------------
'''
def anonymize_dicom_files(source_dir, destination_dir, tags_to_anonymize):
    """
    주어진 DICOM 태그들을 익명화하고 결과를 새로운 경로에 저장합니다.
    각 원본 폴더에 대응하는 새 폴더를 생성하고, 익명화된 파일을 그 안에 저장합니다.

    Parameters:
    source_dir (str): 원본 DICOM 파일이 있는 폴더의 경로.
    destination_dir (str): 익명화된 DICOM 파일을 저장할 새로운 폴더의 경로.
    tags_to_anonymize (list): 익명화할 DICOM 태그 목록.
    """
    for root, dirs, files in os.walk(source_dir):
        for dirname in dirs:
            # 각 하위 폴더에 대응하는 새로운 폴더 경로 생성
            relative_path = os.path.relpath(os.path.join(root, dirname), source_dir)
            new_folder_path = os.path.join(destination_dir, relative_path)
            
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"Created directory: {new_folder_path}")

        for filename in files:
            if filename.lower().endswith('.dcm'):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, source_dir)
                new_folder_path = os.path.join(destination_dir, relative_path)

                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                    print(f"Created directory: {new_folder_path}")

                ds = dcm.dcmread(file_path)

                # 지정된 태그를 익명화
                for tag in tags_to_anonymize:
                    # DICOM 데이터 요소에 접근하기 위한 키 생성
                    for elem in ds:
                        if elem.name in tags_to_anonymize:
                            elem.value = ''

                # 새로운 경로에 익명화된 파일 저장
                new_file_path = os.path.join(new_folder_path, filename)
                ds.save_as(new_file_path)
                print(f"Anonymized and saved: {new_file_path}")


tags_to_anonymize = tags_to_anonymize = [
    "Institution Name",
    "Referring Physician's Name",
    "Station Name",
    "Physician(s) of Record",
    "Attending Physician's Name",
    "Patient's Name",
    "Patient ID",
    "Patient's Birth Date",
    "Patient's Sex",
    "Patient's Age",
    "Patient's Size",
    "Patient's Weight",
    "Study ID",
    "Requesting Physician",
    "Requesting Service"
]

anonymize_dicom_files(new_dir, new_dir, tags_to_anonymize)'''



###-----------------z순서에 맞게 다이콤 솔팅하는 코드---------------------

"""import os
from pathlib import Path
import pydicom
import numpy as np
from shutil import copy2

new_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\anon_data'
new_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\new_data'

'''# new_dir에 patient1부터 patient417까지의 폴더 생성
for i in range(1, 418):
    Path(f"{new_dir}/patient{i}").mkdir(parents=True, exist_ok=True)'''

def sort_and_save_dicom_files(patient_folder, target_folder):

    dicom_files = [f for f in os.listdir(patient_folder) if f.endswith('.dcm')]
    dicom_files_full_path = [os.path.join(patient_folder, f) for f in dicom_files]

    # RT 파일과 그 외 파일을 분류
    rt_files = []
    other_files = []

    for file_path in dicom_files_full_path:
        dicom_data = pydicom.dcmread(file_path, stop_before_pixels=True)
        if hasattr(dicom_data, 'Modality') and dicom_data.Modality == 'RTSTRUCT':
            # RT 파일 분류
            rt_files.append(file_path)
        else:
            other_files.append((file_path, dicom_data.get('ImagePositionPatient', [0,0,0])[2]))

    # ImagePositionPatient[2] 값에 따라 정렬, RT 파일이 아닌 경우만
    dicom_files_sorted = sorted(other_files, key=lambda x: x[1])

    # 정렬된 DICOM 파일과 RT 파일을 새로운 이름으로 저장
    idx = 1
    for dicom_file, _ in dicom_files_sorted:
        ds = pydicom.dcmread(dicom_file)
        new_file_name = f"CT.{idx:03d}.dcm"
        ds.save_as(os.path.join(target_folder, new_file_name))
        idx += 1
    
    # RT 파일 저장
    for rt_file in rt_files:
        ds = pydicom.dcmread(rt_file)
        new_file_name = f"RT.{idx:03d}.dcm"
        ds.save_as(os.path.join(target_folder, new_file_name))
        idx += 1

patients = os.listdir(new_dir)
patients = sorted(patients, key=lambda x : int(x[7:]))
print(patients)


for patient_idx, patient in enumerate(patients):
    if patient_idx > 3:
        data_dir = os.path.join(new_dir, patient)
        save_dir = os.path.join(new_dir, patient)
        
        sort_and_save_dicom_files(data_dir, save_dir)
        print(f"{patient} is done!")"""
        

###------------------------파일 훈련용으로 재정렬------------------------------
import shutil

'''ori_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\nii_data'
new_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\for_train'


patients = os.listdir(ori_dir)
patients = sorted(patients, key=lambda x: int(x[7:]))
labels = os.listdir(new_dir)
labels = sorted(labels, key=lambda x: int(x[5:]))

for idx, patient in enumerate(patients, start=1):
    each_patient  = os.listdir(os.path.join(ori_dir, patient))
    each_patient_path = os.path.join(ori_dir, patient)
    CT = [f for f in each_patient if f.startswith('CT')][0]
    RTs = [f for f in each_patient if f.startswith('RT')]
    
    
    for idx, label in enumerate(labels, start = 1):
        
        if idx == 1:
            each_label_path = os.path.join(new_dir, label)
            files = os.listdir(each_label_path)
            data_dir = [f for f in files if f.startswith('data')]
            data_dir = data_dir[0]
            
            #print(patients)
            #print(data_dir)
            data_path = os.path.join(each_label_path, data_dir)
            imagesTr_path = os.path.join(data_path, 'imagesTr')
            labelsTr_path = os.path.join(data_path, 'labelsTr')
            
            if not os.path.exists(imagesTr_path):
                os.mkdir(imagesTr_path)
            if not os.path.exists(labelsTr_path):
                os.mkdir(labelsTr_path)
            
            for RT in RTs:
                CT_path = os.path.join(each_patient_path, CT)
                new_CT_path = os.path.join(imagesTr_path, CT)
                RT_path = os.path.join(each_patient_path, RT)
                new_RT_path = os.path.join(labelsTr_path, RT)
                
                
                if RT[6:].split('.')[0] == label[5:]:
                    if not os.path.exists(new_CT_path):
                        os.rename(CT_path, new_CT_path)
                        
                    if not os.path.exists(new_RT_path):
                        os.rename(RT_path, new_RT_path)

            
            
            for patient in patients:
                #print(patient)
                patient_path = os.path.join(each_label_path, patient)
                CTnRT = os.listdir(patient_path)
                #print(CTnRT)
                if len(CTnRT) > 1:
                    CT = [f for f in CTnRT if f.startswith('CT')][0]
                    RT = [f for f in CTnRT if f.startswith('RT')][0]

                    CT_path = os.path.join(patient_path, CT)
                    new_CT_path = os.path.join(imagesTr_path, CT)
                    RT_path = os.path.join(patient_path, RT)
                    new_RT_path = os.path.join(labelsTr_path, RT)
                    os.rename(CT_path, new_CT_path)
                    os.rename(RT_path, new_RT_path)
            
            for patient in patients:
                patient_path = os.path.join(each_label_path, patient)
                CTnRT = os.listdir(patient_path)
                if len(CTnRT) == 0:
                    os.rmdir(patient_path)'''
                    
                    
                    
import os
import shutil

ori_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\nii_data'
new_dir = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\for_train'


patients = os.listdir(ori_dir)
patients.sort(key=lambda x: int(x.replace("patient", "")))

for patient in patients:
    files = os.path.join(ori_dir, patient)
    lst = os.listdir(files)
    RT = [f for f in lst if f.startswith('RTst') & f.endswith('-1.nii.gz')]
    if len(RT) == 0:
        print(patient)


'''patients = os.listdir(ori_dir)
patients.sort(key=lambda x: int(x.replace("patient", "")))

# label1 경로 설정
label_dir = os.path.join(new_dir, 'label1', 'data')
imagesTr_path = os.path.join(label_dir, 'imagesTr')
labelsTr_path = os.path.join(label_dir, 'labelsTr')

# 필요한 경우 디렉토리 생성
if not os.path.exists(imagesTr_path):
    os.makedirs(imagesTr_path)
if not os.path.exists(labelsTr_path):
    os.makedirs(labelsTr_path)

for patient in patients:
    patient_dir = os.path.join(ori_dir, patient)
    files = os.listdir(patient_dir)
    
    # CT 파일 찾기
    ct_files = [file for file in files if file.startswith("CT")]
    # RT 파일 찾기
    rt_files = [file for file in files if file.startswith("RTst")]
    
    for ct_file in ct_files:
        ct_path = os.path.join(patient_dir, ct_file)
        # CT 파일을 imagesTr로 복사
        shutil.copy(ct_path, os.path.join(imagesTr_path, f"{patient}_{ct_file}"))
        
    for rt_file in rt_files:
        rt_number = rt_file.split('-')[1].split('.')[0]  # RT 파일의 순번 추출
        rt_path = os.path.join(patient_dir, rt_file)
        # 해당하는 RT 파일을 labelsTr로 복사
        # RT 파일 이름에서 순번을 분석하여 label 번호와 일치하는 경우에만 복사
        if rt_number == '1':  # 여기서 '1'은 label1에 해당하는 RT 파일을 의미
            shutil.copy(rt_path, os.path.join(labelsTr_path, f"{patient}_{rt_file}"))'''
import os

# 파일 이름에서 "patientX_" 부분을 제거
def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith("patient"):
            new_filename = "_".join(filename.split("_")[1:])  # 첫 번째 "_" 이후의 부분을 파일 이름으로 사용
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


imagesTr_path = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\for_train\\label1\\data\\imagesTr'
labelsTr_path = 'C:\\Users\\slicemind1\\Documents\\AutoSegmentation\\for_train\\label1\\data\\labelsTr'

#rename_files(imagesTr_path)
#rename_files(labelsTr_path)

# CT와 RT 파일 pair 확인