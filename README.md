# PCVch08

python计算机视觉第八章实验

## 图像分类

所谓图像分类问题，就是已有固定的分类标签集合，然后对于输入的图像，从分类标签集合中找出一个分类标签，最后把分类标签分配给该输入图像。虽然看起来挺简单的，但这可是计算机视觉领域的核心问题之一,计算机视觉领域中很多看似不同的问题（比如物体检测和分割），都可以被归结为图像分类问题。

### KNN

K最近邻(k-Nearest Neighbor，KNN)分类算法，是简单的机器学习算法之一。这种方法的流程思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。 KNN方法虽然从原理上也依赖于极限定理，但在类别决策时，只与极少量的相邻样本有关。由于KNN方法主要靠周围有限的邻近的样本，而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，KNN方法较其他方法更为适合。

简单的理解，我有一组数据，比如每个数据都是n维向量，那么我们可以在n维空间表示这个数据，这些数据都有对应的标签值，也就是我们感兴趣的预测变量。那么当我们接到一个新的数据的时候，我们可以计算这个新数据和我们已知的训练数据之间的距离，找出其中最近的k个数据，对这k个数据对应的标签值取平均值就是我们得出的预测值。简单粗暴，谁离我近，就认为谁能代表我，我就用你们的属性作为我的属性。

**存在问题**：k值得选取对kNN学习模型有着很大的影响。若k值过小，预测结果会对噪音样本点显得异常敏感。特别地，当k等于1时，kNN退化成最近邻算法，没有了显式的学习过程。若k值过大，会有较大的邻域训练样本进行预测，可以减小噪音样本点的减少；但是距离较远的训练样本点对预测结果会有贡献，以至于造成预测结果错误。

在此次实验中实现的是一个最基本的KNN。给定训练样本集和对应的标记列表。这些训练样本集和对应的标记可以在一个数组中成行摆放或者干脆摆放在列表里，训练样本可能是数字，字符串等你任何喜欢的形状。如下代码实现了一个基本的KNN分类器：

```python
from numpy import * 

class KnnClassifier(object):
    
    def __init__(self,labels,samples):
        """ Initialize classifier with training data. """
        
        self.labels = labels
        self.samples = samples
    
    def classify(self,point,k=3):
        """ Classify a point against k nearest 
            in the training data, return label. """
        
        # compute distance to all training points
        dist = array([L2dist(point,s) for s in self.samples])
        
        # sort them
        ndx = dist.argsort()
        
        # use dictionary to store the k nearest
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label,0)
            votes[label] += 1
            
        return max(votes, key=lambda x: votes.get(x))


def L2dist(p1,p2):
    return sqrt( sum( (p1-p2)**2) )

def L1dist(v1,v2):
    return sum(abs(v1-v2))
```

现在我将建立一个简单的二维示例数据集来说明并可视化分类器的工作原理：

```python
# -*- coding: utf-8 -*-
from numpy.random import randn
import pickle
from pylab import *

# create sample data of 2D points
n = 200
# two normal distributions
class_1 = 0.6 * randn(n,2)
class_2 = 1.2 * randn(n,2) + array([5,1])
labels = hstack((ones(n),-ones(n)))
# save with Pickle
#with open('points_normal.pkl', 'w') as f:
with open('points_normal_test.pkl', 'wb') as f:
    pickle.dump(class_1,f)
    pickle.dump(class_2,f)
    pickle.dump(labels,f)
# normal distribution and ring around it
print ("save OK!")
class_1 = 0.6 * randn(n,2)
r = 0.8 * randn(n,1) + 5
angle = 2*pi * randn(n,1)
class_2 = hstack((r*cos(angle),r*sin(angle)))
labels = hstack((ones(n),-ones(n)))
# save with Pickle
#with open('points_ring.pkl', 'w') as f:
with open('points_ring_test.pkl', 'wb') as f:
    pickle.dump(class_1,f)
    pickle.dump(class_2,f)
    pickle.dump(labels,f)
    
print ("save OK!")
```

如上代码创建了两个随机的不同的二维点集，用Pickle模块来保存创建的数据，这两类数据我们将其中一个用来当成训练数据集，另一个则当成测试数据集。

那么我们接下来看看怎么用KNN分类器完成，如下代码：

```python
# -*- coding: utf-8 -*-
import pickle
from pylab import *
from PCV.classifiers import knn
from PCV.tools import imtools

pklist=['points_normal.pkl','points_ring.pkl']

figure()

# load 2D points using Pickle
for i, pklfile in enumerate(pklist):
    with open(pklfile, 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    # load test data using Pickle
    with open(pklfile[:-4]+'_test.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)

    model = knn.KnnClassifier(labels,vstack((class_1,class_2)))
    # test on the first point
    print (model.classify(class_1[0]))

    #define function for plotting
    def classify(x,y,model=model):
        return array([model.classify([xx,yy]) for (xx,yy) in zip(x,y)])

    # lot the classification boundary
    subplot(1,2,i+1)
    imtools.plot_2D_boundary([-6,6,-6,6],[class_1,class_2],classify,[1,-1])
    titlename=pklfile[:-4]
    title(titlename)
savefig("test.png")
show()

```

这里用Pickle模块来创建一个KNN分类器模型

**显示结果**：

![](https://github.com/zengqq1997/PCVch08/blob/master/result１.jpg)

### DSIFT

现在我们已经知道了分类器的使用原理了，但是图像是如何进行分类的呢？要对图像进行分类，我们需要一个特征向量来表示一幅图像。在以前实验中我们用SIFT来作为图像的特征向量，那么今天我将介绍另一种表示形式，即稠密SIFT特征向量。

**DSIFT和SIFT的区别**：图像分类问题大多用Dense-SIFT。图像检索总是用SIFT（利用了检测子）。

DSIFT应用代码如下：

```python
from PIL import Image
from numpy import *
import os

from PCV.localdescriptors import sift


def process_image_dsift(imagename,resultname,size=20,steps=10,force_orientation=False,resize=None):
    """ Process an image with densely sampled SIFT descriptors 
        and save the results in a file. Optional input: size of features, 
        steps between locations, forcing computation of descriptor orientation 
        (False means all are oriented upwards), tuple for resizing the image."""

    im = Image.open(imagename).convert('L')
    if resize!=None:
        im = im.resize(resize)
    m,n = im.size
    
    if imagename[-3:] != 'pgm':
        #create a pgm file
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # create frames and save to temporary file
    scale = size/3.0
    x,y = meshgrid(range(steps,m,steps),range(steps,n,steps))
    xx,yy = x.flatten(),y.flatten()
    frame = array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])])
    savetxt('tmp.frame',frame.T,fmt='%03.3f')
    
    path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
    path = path + "\\python2-ch08\\win32vlfeat\\sift.exe "
    if force_orientation:
        cmmd = str(path+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame --orientations")
    else:
        cmmd = str(path+imagename+" --output="+resultname+
                    " --read-frames=tmp.frame")
    os.system(cmmd)
    print ('processed', imagename, 'to', resultname)


# -*- coding: utf-8 -*-
from PCV.localdescriptors import sift, dsift
from pylab import  *
from PIL import Image

dsift.process_image_dsift('gesture/empire.jpg','empire.dsift',90,40,True)
l,d = sift.read_features_from_file('empire.dsift')
im = array(Image.open('gesture/empire.jpg'))
sift.plot_features(im,l,True)
title('dense SIFT')
show()
```

**显示结果**：

![](https://github.com/zengqq1997/PCVch08/blob/master/result２.jpg)

### 手势识别

在了解过分类器的原理和如何进行分类后，我将开始进行实际的操作，此次实验将进行手势图像的识别。首先我们静态手势数据集进行演示。将所有图像分为两类，并分别放入名为train和test的文件夹中。

刚才我们已经用稠密ＳＩＦＴ对图像进行过演示了，可以得到所有图像的特征向量。这里，再次假设列表ｉｍｌｉｓｔ中包含了所有的图像的文件名，可以通过下列代码得到每幅图的稠密ｓｉｆｔ特征：



```python
# -*- coding: utf-8 -*-
import os
from PCV.localdescriptors import sift, dsift
from pylab import  *
from PIL import Image

imlist=['gesture/imgs/a-uniform36.jpg','gesture/imgs/c-uniform01.jpg',
        'gesture/imgs/f-uniform36.jpg','gesture/imgs/liu-uniform31.jpg',
       'gesture/imgs/ye-uniform17.jpg','gesture/imgs/yi-uniform12.jpg']



figure()
for i, im in enumerate(imlist):
    print (im)
    dsift.process_image_dsift(im,im[:-3]+'dsift',10,5,True,resize=(50,50))
    l,d = sift.read_features_from_file(im[:-3]+'dsift')
    dirpath, filename=os.path.split(im)
    im = array(Image.open(im))
    #显示手势含义title
    titlename=filename[:-14]
    subplot(2,3,i+1)
    sift.plot_features(im,l,True)
    title(titlename)
show()
`
```

**显示结果**：

![](https://github.com/zengqq1997/PCVch08/blob/master/result３.jpg)

接下来我们可以读取我们刚才准备好的训练数据集、测试数据集来进行特征的标定

首先我们将训练集及其标记作为输入，创建分类器对象；然后，我们在整个测试集上遍历并用ｃｌａｓｓｉｆｙ（）方法对每幅图进行分类。将布尔数组和１相乘并求和，可以计算分类的正确率。

虽然正确率显示对于一给定的测试有多少图像是正确的，但是他并没告诉我们哪些手势难以分类。这里引用了混淆矩阵，这是一个可以显示每类有多少个样本被分类在一类矩阵中，可以显示错误的分布情况，以及哪些类是经常相互混淆的

代码如下：

```python
# -*- coding: utf-8 -*-
from PCV.localdescriptors import dsift
import os
from PCV.localdescriptors import sift
from pylab import *
from PCV.classifiers import knn

def get_imagelist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def read_gesture_features_labels(path):
    # create list of all files ending in .dsift
    featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]
    # read the features
    features = []
    for featfile in featlist:
        l,d = sift.read_features_from_file(featfile)
        features.append(d.flatten())
    features = array(features)
    # create labels
    labels = [featfile.split('/')[-1][0] for featfile in featlist]
    return features,array(labels)

def print_confusion(res,labels,classnames):
    n = len(classnames)
    # confusion matrix
    class_ind = dict([(classnames[i],i) for i in range(n)])
    confuse = zeros((n,n))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]],class_ind[test_labels[i]]] += 1
    print ('Confusion matrix for')
    print (classnames)
    print (confuse)

filelist_train = get_imagelist('gesture/train1')
filelist_test = get_imagelist('gesture/test1')
imlist=filelist_train+filelist_test

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))

features,labels = read_gesture_features_labels('gesture/train1/')
test_features,test_labels = read_gesture_features_labels('gesture/test1/')
classnames = unique(labels)

# test kNN
k = 1
knn_classifier = knn.KnnClassifier(labels,features)
res = array([knn_classifier.classify(test_features[i],k) for i in
range(len(test_labels))])
# accuracy
acc = sum(1.0*(res==test_labels)) / len(test_labels)
print ('Accuracy:', acc)

print_confusion(res,test_labels,classnames)

```

------

**显示结果**：

![](https://github.com/zengqq1997/PCVch08/blob/master/result４.jpg)

如图显示正确率为１，目前混淆矩阵并没有经常混淆的类

### 小结

由于训练数据集不够庞大导致最后的实验结果不是十分尽人意，在接下去的要增加数据量
