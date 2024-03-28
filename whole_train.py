import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    ResizeWithPadOrCropd
)
from monai.transforms import Transform

from monai.config import print_config
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch

import logging
logging.getLogger("monai").setLevel(logging.ERROR)  # MONAI 로거의 레벨을 'ERROR'로 설정

import warnings
warnings.filterwarnings("ignore") # 경고제거용
#print_config()
import numpy as np



ori_dir = 'C:\\Users\\guja1\\Documents\\AutoSegmentation\\for_train'
labels = sorted(os.listdir(ori_dir), key=lambda x: int(x[5:]))

adaptive_scale={"label1": [-100, 100], "label2": [-100, 100], "label3": [-50, 100], "label4": [-1000, -700], "label5": [30, 45], "label6": [30, 45], "label7": [40, 100], "label8": [-10, 10], "label9": [40, 60], "label10": [60, 100], "label11": [0, 100], "label12": [0, 100], "label13": [45, 65], "label14": [-10, 20], "label15": [30, 50], "label16": [-950, -700], "label17": [-950, -700], "label18": [40, 80], "label19": [-100, 100], "label20": [-100, 100], "label21": [-100, 100], "label22": [-100, 100], "label23": [1000, 3000], "label24": [-950, -700], "label25": [-300, -100], "label26": [20, 40]}
adaptive_size=(96,96,96)
for num, label in enumerate(labels, start=1):
    if label!="label1" and label != "label25": 
        root_dir = os.path.join(ori_dir, label)
        print(root_dir)
        rand_num = np.random.randint(10)
        print(rand_num)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() == True:
            print('cuda is available')

        import monai
        print(monai.__version__)
        ###------------------------------------------Transforms------------------------------------------
        num_samples = 2

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=float(adaptive_scale[label][0]),
                    a_max=float(adaptive_scale[label][1]),
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    #slice_thickness=3
                    pixdim=(1.0, 1.0, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=adaptive_size,
                    pos=1,
                    neg=1,
                    num_samples=num_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.10,
                ),
                ResizeWithPadOrCropd(keys=["image", "label"],
                spatial_size=adaptive_size,
                mode='constant')
            ]
        )


        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"],
                    #a_min=-175,
                    #a_max=250,
                    #실험적으로 수정
                    #a_min=-100,
                    #a_max=200,
                    a_min=float(adaptive_scale[label][0]),
                    a_max=float(adaptive_scale[label][1]),
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
            ]
        )





        ###---------------------------------------pixdim error solving------------------------------
        import nibabel as nib
        from monai.data import partition_dataset
        nib.imageglobals.logger.setLevel(40)


        data_dir = f"C:\\Users\\guja1\\Documents\\AutoSegmentation\\for_train\\{label}\\data\\"
        split_json = "dataset_1.json"

        datasets = data_dir + split_json

        datalist = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")
        
        train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=12,
            cache_rate=1.0,
            num_workers=8
        )
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)

        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num=6,
            cache_rate=1.0,
            num_workers=4
        )
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        # 테스트 데이터셋 및 데이터 로더



        set_track_meta(False)

        ###---------------------confirm transformed data----------------------------

        import os
        import nibabel as nib
        import numpy as np


        ###-------------------------------------------Modeling------------------------------------------

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SwinUNETR(
            img_size=adaptive_size,
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,).to(device)

        torch.backends.cudnn.benchmark = True

        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)

        scaler = torch.cuda.amp.GradScaler()


        ###------------------------------------Train/Validation define-------------------------------------------

        def validation(epoch_iterator_val, last_fc_size):
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for batch in epoch_iterator_val:
                    val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                    if val_labels.max() >= last_fc_size:
                        val_labels[val_labels >= last_fc_size] = 0
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_inputs, adaptive_size, 1, model)
                        #val_outputs = model(val_inputs)
                    val_labels_list = decollate_batch(val_labels)
                    val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                    val_outputs_list = decollate_batch(val_outputs)
                    val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                    if val_output_convert is not None and val_labels_convert is not None:
                        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                        surface_distance_metric(y_pred=val_output_convert, y=val_labels_convert)
                        iou_metric(y_pred=val_output_convert, y=val_labels_convert)
                    else:
                        print("Invalid data encountered in validation")
                        continue
                    epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
                mean_dice_val = dice_metric.aggregate().item()
                
                mean_asd_val = surface_distance_metric.aggregate().item()
                mean_iou_val = iou_metric.aggregate().item()
                dice_metric.reset()
                
                surface_distance_metric.reset()
                iou_metric.reset()

            return mean_dice_val,  mean_asd_val, mean_iou_val



        def train(global_step, train_loader, dice_val_best, global_step_best, last_fc_size):
            model.train()
            epoch_loss = 0
            step = 0
            epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
            for step, batch in enumerate(epoch_iterator):
                step += 1
                x, y = (batch["image"].cuda(), batch["label"].cuda())
                
                if y.max() >= last_fc_size:
                    print("\nReplace labels >= {} to 0".format(last_fc_size))
                    y[y >= last_fc_size] = 0
                
                with torch.cuda.amp.autocast():
                    logit_map = model(x)
                    if logit_map is not None and y is not None:
                        
                        loss = loss_function(logit_map, y)
                    else:
                        print("Invalid data encountered in training")
                        continue  # 다음 배치로 넘어감
                scaler.scale(loss).backward()


                total_norm=0
                for p in model.parameters():
                    
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                if global_step%500 == 0:
                    
                    print(total_norm)
                epoch_loss += loss.item()
                
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()
                

                epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
                
                if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                    dice_val, asd_val, iou_val = validation(epoch_iterator_val, last_fc_size)
                    epoch_loss /= step

                    epoch_loss_values.append(epoch_loss)
                    metric_values.append(dice_val)
                    
                    asd_values.append(asd_val)
                    iou_values.append(iou_val)

                    if dice_val > dice_val_best:
                        dice_val_best = dice_val
                        global_step_best = global_step
                        torch.save(model.state_dict(), os.path.join(root_dir, f"{rand_num} best_metric_model.pth"))
                        print(
                            "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} \n  asd val: {} iou val:{}".format(dice_val_best, dice_val, asd_val, iou_val)
                        )
                    else:
                        print(
                            "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} \n asd val: {} iou val:{}".format(dice_val_best, dice_val, asd_val, iou_val)
                        )
            
                global_step += 1
            return global_step, dice_val_best, global_step_best


        ###------------------------------------Training---------------------------------------
        last_fc_size = 2
        max_iterations = 30000
        eval_num = 500
        post_label = AsDiscrete(to_onehot=2)
        post_pred = AsDiscrete(argmax=True, to_onehot=2)
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        surface_distance_metric = SurfaceDistanceMetric(include_background=True)
        iou_metric = MeanIoU(include_background=True)

        global_step = 0
        dice_val_best = 0.0
        global_step_best = 0  


        epoch_loss_values = []
        metric_values = []
        hd_values = []
        asd_values = []
        iou_values = []


        while global_step < max_iterations:
            global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best, last_fc_size)
        print('epoch_loss_values:',epoch_loss_values, 'metric_values:',metric_values)

        import json

        # Data to be saved
        data = {
            "epoch_loss_values": epoch_loss_values,
            "metric_values": metric_values,
            "asd_values": asd_values,
            "iou_values": iou_values,
        }

        # Convert to JSON string
        json_data = json.dumps(data, indent=4)

        # Write to file
        with open(f'C:\\Users\\guja1\\Documents\\AutoSegmentation\\for_train\\{label}\\data\\metrics.json', 'w') as file:
            file.write(json_data)