
# Usage


## pth --> onnx
Convert `pth` to `onnx`:
```bash
python pth2onnx.py --weights ./saved_model/plate_rec_color.pth \
                --save_path ./saved_model/plate_rec_color_bs2.onnx  \
                --batch_size 2 \
                --simplify
```


## onnx --> rknn
Convert `onnx` to `rknn`:
```bash
python rknn_transfer_batch.py --onnx ./saved_model/plate_rec_color_bs1.onnx \
                --save_path ./saved_model/plate_rec_color_bs1.rknn
```

