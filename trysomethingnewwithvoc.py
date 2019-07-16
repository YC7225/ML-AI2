voc_2007_images_dir = "../input/pascal-voc-2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"
voc_2012_images_dir = "../input/pascal-voc-2012/VOC2012/VOC2012/JPEGImages/"
voc_2007_annotations_dir = "../input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/"
voc_2012_annotations_dir = "../input/pascal-voc-2012/VOC2012/VOC2012/Annotations/"

voc_2007_train_image = "../input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/train.txt"
voc_2012_train_image = "../input/pascal-voc-2012/VOC2012/VOC2012/ImageSets/Main/train.txt"
voc_2007_val_image   = "../input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/val.txt"
voc_2012_val_image   = "../input/pascal-voc-2012/VOC2012/VOC2012/ImageSets/Main/val.txt"
voc_2007_trainval_image = "../input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"
voc_2012_trainval_image = "../input/pascal-voc-2012/VOC2012/VOC2012/ImageSets/Main/trainval.txt"
voc_2007_test_image = "../input/pascal-voc-2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt"

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

train_dataset = train_dataset.parse_xml(images_dirs=[voc_2007_images_dir,
                                     voc_2012_images_dir],
                        image_set_filenames=[voc_2007_trainval_image,
                                             voc_2012_trainval_image],
                        annotations_dirs=[voc_2007_annotations_dir,
                                          voc_2012_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[voc_2007_images_dir],
                      image_set_filenames=[voc_2007_test_image],
                      annotations_dirs=[voc_2007_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)