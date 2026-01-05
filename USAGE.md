

Notice!!!!!!!!!!!!!!!!!!!!!!!!!!!!: This project is not the newest, get newest from gitee.










### Install dependence
```bash
pip install tensorboardX
```


---

### Generate dataset labels:
```bash
python plateLabel.py --image_path /home/david/dataset/plate/generate/train --label_file datasets/train.txt

python plateLabel.py --image_path /home/david/dataset/plate/generate/val --label_file datasets/val.txt
```


---

### Train 

Train plate recognition:
```bash
export CUDA_VISIBLE_DEVICES=0;nohup python train.py --cfg lib/config/360CC_config.yaml > logs/plate_small_log.txt 2>&1 &

cp output/360CC/crnn/2026-01-04-14-51/checkpoints/checkpoint_99_acc_0.9085.pth saved_model/plate_rec_bj.pth
```

Add color recognition branch:
```bash
python train_fix_color.py --weights saved_model/plate_rec_bj.pth --train_path datasets/train  --val_path datasets/val --model_path color_model
```
The output in `./color_model`

----

### Pth inference
```bash
python demo_plate_color.py --model_path saved_model/plate_rec_color.pth --image_path images/test.jpg


python demo.py --model_path output/360CC/crnn/2026-01-04-14-51/checkpoints/checkpoint_99_acc_0.9085.pth --image_path images/tmp6C79.png
```

---

### Export rknn
In `exprot/rknn` directory.


---

### Export onnx
```bash
python export.py --weights saved_model/plate_rec_color.pth --save_path saved_model/plate_rec_color.onnx  --simplify
```

---

### Onnx inference
```bash
python onnx_infer.py --onnx_file saved_model/plate_rec_color.onnx  --image_path images/test.jpg
```

