# Download COCO validation dataset

Run the download script from the repository root:

```bash
./scripts/download_coco_val.sh
```

The script performs the following steps:

```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

mv val2017 images

rm val2017.zip
rm annotations_trainval2017.zip
```
