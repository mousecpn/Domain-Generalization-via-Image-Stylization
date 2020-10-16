## Frustratingly Simple Domain Generalization via Image Stylization

This is an unofficial PyTorch implementation of Frustratingly Simple Domain Generalization via Image Stylization.

[[arxiv]](https://arxiv.org/pdf/2006.11207.pdf)

Some of the code is from https://github.com/fmcarlucci/JigenDG and https://github.com/xunhuang1995/AdaIN-style .

#### data

PACS: https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk

Download the raw images. It is recommended to symlink the dataset root to `data`.

```
Domain-Generalization-via-Image-Stylization
├── data
│   ├── PACS
│   |   ├── kfold
│   |   |   ├── art_painting
│   |   |   ├── cartoon
│   |   |   ├── photo
│   |   |   ├── sketch
```



#### train

`python main.py --batch_size 16 --n_classes 7 --learning_rate 0.001 --val_size 0.1 --folder_name test --train_all True --TTA False --image_size 256 --nesterov True --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --source art_painting cartoon photo --target sketch --bias_whole_image 0.7`

