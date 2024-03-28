### WholeCTSeg: 전신 CT Segmentation 프로젝트

Whole Body CT를 각 장기의 사이즈에따라 adaptive하게 crop & hu_clipping을 진행

이후 whole_train.py 파일을 통해 한번에 Segmentation 수행

해당 .pth 파일들은 추후 GUI에서 if문에 걸려 사용자의 클릭에 따라 추론을 진행하게 됨
