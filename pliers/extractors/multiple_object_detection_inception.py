import contextlib
import sys
import io
from pliers.stimuli import ImageStim
from pliers.extractors import TensorFlowInceptionV3Extractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from copy import *

'''
Silence the stdout of a function
'''
@contextlib.contextmanager
def nostdoutstderr():
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout =  io.BytesIO()
    sys.stderr = io.BytesIO()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr


'''
this function returns all possible bounding boxes
that span the image with different steps and strides. 
The size of the return is nb_bbxs*4.
The bounding box is written under this format [xmin, ymin, xmax, ymax].
'''
def generate_bb(im_path):
    alpha = 1.5
    stride = 5
    bbx = list()
    img = ImageStim(im_path)
    height= (img.data).shape[0]
    width= (img.data).shape[1]
    h_step = stride
    w_step = stride
    while h_step < height and w_step < width:
        for i in range(0,height-h_step,stride):
            for j in range(0,width-w_step,stride):
                bbx.append([i, j, i+h_step, j+w_step])
        w_step = int(stride*alpha)
        h_step = int(stride*alpha)
        stride = int(stride*alpha)

    return np.array(bbx)


def generate_scores(im_path,bbxs):
    img = ImageStim(im_path)
    d=img.data
    ext=TensorFlowInceptionV3Extractor(model_dir='../',data_url=None,num_predictions=1)
    nb=len(bbxs)
    scores=[]
    with nostdoutstderr():
        for i in range(0,nb):
            xmin,ymin,xmax,ymax=bbxs[i,:]
            sample=ImageStim(data=d[xmin:xmax,ymin:ymax,:])        
            result=ext.transform(sample)
            scores.append(result._data[0])
    #scores.append(result.features[0])
    return scores

def overlap_ratio(bbx1,bbx2):
    xmin1,ymin1,xmax1,ymax1=bbx1
    s1=(xmax1-xmin1)*(ymax1-ymin1)
    xmin2,ymin2,xmax2,ymax2=bbx2
    s2=(xmax2-xmin2)*(ymax2-ymin2)

    if (xmax1<=xmin2) or (xmax2<=xmin1) or (ymax1<=ymin2) or (ymax2<=ymin1):
        return 0
    else:
        s=abs(min(xmax1,xmax2)-max(xmin1,xmin2))*abs(min(ymax1,ymax2)-max(ymin1,ymin2))
        return max(s/s1,s/s2)

def final_detection(bbxs,scores,thresh):
    idxs = sorted(range(len(scores)), key=scores.__getitem__)
    idxs.reverse()
    bb = []
    sc=[]
    for i in range (0,len(idxs)):
        bb.append(bbxs[idxs[i]])
        sc.append(scores[idxs[i]])
    #sort bb according to idxs
    l = []
    s=[]
    while len(bb) != 0:
        l.append(bb[0])
        s.append(sc[0])
        bb = bb[1:]
        sc = sc[1:]
        b = copy(bb)
        ss = copy(sc)
        j = 0
        for i in range(len(bb)):
            if overlap_ratio(l[-1],bb[i]) > thresh:
                del b[j]
                del ss[j]
            else:
                j += 1
        bb = b
        sc = ss

    return l, s


if __name__=="__main__":
    im_path="./z.jpg"
    #img = image.load_img(im_path, target_size=(224, 224))
    img = ImageStim(im_path)
    bbxs=generate_bb(im_path)
    #print(bbxs[0])
    # #bbxs=np.array([[0,0,(img.data).shape[0]-10,(img.data).shape[1]-10]])
    S=generate_scores(im_path,bbxs)
    #print(max(S))
    # #S.sort()
    #print(S[666])

    fig,ax = plt.subplots(1)
    ax.imshow(img.data)
    # # rect = patches.Rectangle((bbxs[24000,1],bbxs[24000,0]),bbxs[24000,3]-bbxs[24000,1],bbxs[24000,2]-bbxs[24000,0],linewidth=1,edgecolor='r',facecolor='none')
    #rect = patches.Rectangle((0,0),220,220,linewidth=1,edgecolor='r',facecolor='none')
    #
    #
    #
    # # # Add the patch to the Axes
    #
    #
    
    #xmin,ymin,xmax,ymax=bbxs[666,:]

    #xmin,ymin,xmax,ymax=[2,100,54,140]
    # # #print(xmin,'*****',xmax,'*****',ymin,'*****',ymax)
    #ext=TensorFlowInceptionV3Extractor(model_dir='../',data_url=None,num_predictions=1)
    #x = image.img_to_array(img)
    #sample=ImageStim(data=img.data[xmin:xmax,ymin:ymax,:])
    #result=ext.transform(sample)
    #print(result.features)
    #plt.imshow(sample.data)
    #plt.show()

    #bbx1=[10,5,30,20]
    #bbx2=[10,3,15,15]
    #s=overlap_ratio(bbx1,bbx2)
    #bbxs=[bbx1,bbx2]
    #scores=[0.8,0.95]
    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    l, s=final_detection(bbxs,S,0.3)
    ind=[i for i in range(len(s)) if s[i][0]>=0.75]
    bbxs1 = [bbxs[i,:] for i in ind]
    #s1 = [s[i][0] for i in ind]
    for i in l:
        xmin,ymin,xmax,ymax = i
        rect = patches.Rectangle((ymin,xmin),ymax-ymin,xmax-xmin,linewidth=1,edgecolor=color[np.random.randint(0,8)],facecolor='none')
        ax.add_patch(rect)
        ext=TensorFlowInceptionV3Extractor(model_dir='../',data_url=None,num_predictions=1)
        #x = image.img_to_array(img)
        sample=ImageStim(data=img.data[xmin:xmax,ymin:ymax,:])
        result=ext.transform(sample)
        print(result.features)
    plt.show()
    #print('final detection: ',l)

