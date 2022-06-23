这个项目是将SFCN-OPI的代码模型用在了pannuke数据集上。
除了sfcn-opi模型，还添加了文泰的细检测部分，
并且数据集采用了下采样到20x的方式


mask粗检测部分还是用的sfcn-opi的mask
细检测和分类部分就是用的文泰的repel——code编码