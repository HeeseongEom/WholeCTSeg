## WholeCTSeg: 전신 CT Segmentation 프로젝트

WholeCTSeg 프로젝트는 전신 CT 이미지를 대상으로 각 장기별 사이즈에 적응적으로 크롭 및 Hounsfield 단위 클리핑을 수행한 후, 세그멘테이션을 자동으로 진행하는 것을 목표로 합니다. 이 과정을 통해 생성된 모델 파일(.pth)은 GUI에서 사용자의 입력에 따라 조건부로 추론을 실행하게 됩니다.

### 주요 특징

- **Adaptive Cropping & HU Clipping**: Whole Body CT 이미지를 각 장기의 사이즈에 따라 적응적으로 크롭하고 Hounsfield 단위로 클리핑을 진행합니다.
  
- **일괄 Segmentation 수행**: `whole_train.py` 파일을 사용하여 전체 Segmentation 과정을 한 번에 수행합니다.
  
- **조건부 추론 실행**: 생성된 .pth 파일들은 GUI에서 if문 조건에 따라 사용자의 클릭에 반응하여 추론을 진행합니다.
