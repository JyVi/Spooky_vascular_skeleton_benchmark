# CSV:

timestamp | config file name | skeleton technique name | iteration num | params | file name in | parents folder name | dataset name | annotation type (last folder name) | image out name | excution time dureation seconds
YYYY-MM-DD HH-MM-SS | str | str | int | { json compressed } | str | str | str | str | str | float

2025-10-21 12-12-12 | config_001_Lisa.yml | Lisa | 23 | {} | 12 | gt/FAZ | Rose-O | FAZ | Lisa_23_gt_FAZ_Rose-O.png | 0.01

# Dataset naming convention :

Dataset
├── Sub dataset name
│   ├── 1st kind of image
│   │   ├── image_name.png/.tif/.jpg
│   │   ├── ...
│   │   └── image_name.png/.tif/.jpg
│   └── Nth kind of image
├── ...
│   └── ...
└── Sub dataset name
    ├── 1st kind of image
    ├── ...
    └── Nth kind of image

# Results:

Master Output Folder/batchNth
├── batch0
│   ├── config_000_Method_Method_iter00
│   │   ├── Method_image_name_nth.png
│   │   ├── ...
│   │   └── Method_image_name_nth.png
│   ├── ...
│   └── config_999_Method_Method_iter99
├── batch1
│   ├── config_000_Method_Method_iter00
│   ├── ...
│   └── config_999_Method_Method_iter99
├── ...
└── batchNth
    ├── config_000_Method_Method_iter00
    ├── ...
    └── config_999_Method_Method_iter99




The complete current dataset architecture:
Chase-DB1
└── 2MBs
    ├── gt
    └── img
DRIVE
└── DRIVE
    ├── gt
    ├── images
    └── mask
HRF
└── HRF
    ├── gt
    ├── images
    └── mask
ROSE
├── Backup_zip_files
├── ROSE-1
│   ├── DVC
│   │   ├── gt
│   │   └── img
│   ├── SVC
│   │   ├── gt
│   │   ├── img
│   │   ├── thick_gt
│   │   └── thin_gt
│   └── SVC_DVC
│       ├── gt
│       ├── img
│       ├── thick_gt
│       └── thin_gt
├── ROSE-2
│   ├── gt
│   └── original
└── ROSE-O
    ├── gt
    │   ├── FAZ
    │   ├── junctions
    │   └── vessel
    │       ├── DCC
    │       ├── SCC
    │       └── WRCC
    └── img
        ├── DVC
        ├── IVC
        └── SVC
STARE
├── annotations
├── RawImages
├── STARE
│   ├── gt
│   │   ├── labels-ah
│   │   ├── labels-vk
│   │   └── results_PAPER
│   └── images
└── STARE_GT_IMAGES



