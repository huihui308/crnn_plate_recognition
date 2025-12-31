

Convert `pth` to `onnx`:
```bash
python export.py --weights ./../saved_model/plate_rec_color.pth --save_path saved_model/plate_rec_color.onnx  --simplify
```

---


Convert `onnx` to `rknn`:
```bash
python rknn_transfer.py
```













