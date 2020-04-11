import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
from PIL import Image,ImageOps
from tqdm import tqdm
import joblib
BASE="../input/CUB_200_2011"

SAVEPATH="../input/pickles"
SEGPATH="../input/segmentations"

RESIZE=(256,256)
MASK=True

bbox=pd.read_csv( os.path.join(BASE,"bounding_boxes.txt")\
                                ,names=["id","x","y","w","h"],sep=" ").set_index("id")
imgpaths=pd.read_csv(os.path.join(BASE,"images.txt"),\
                            header=None,sep=" ",names=["id","path"]).set_index("id")
def processImage(path,filepath):
    rawimage=plt.imread(path)


    mask=plt.imread(os.path.join(SEGPATH,filepath[:-3]+"png"))

    # plt.imshow(mask)
    if MASK:
        maskedIm=np.array(Image.fromarray(mask).convert('RGB'))*rawimage
    else:
        maskedIm=rawimage
    # plt.imshow(maskedIm)
    row=bbox.loc[idx]
    row=row.astype(int)
    x,y,w,h=row["x"],row["y"],row["w"],row["h"]
    cropped=maskedIm[y:y+h,x:x+w,:]
    edge=max(cropped.shape[0],cropped.shape[1])
    # print(edge)
    padtopbot=(edge-cropped.shape[0])//2
    padlr=(edge-cropped.shape[1])//2
    padding=(padlr,padtopbot,padlr,padtopbot)
    # left,top,right,bottom

    new_im = ImageOps.expand(Image.fromarray(cropped), padding).resize(RESIZE)

    return new_im


if __name__ == "__main__":
    
    
    
    
    image_ids=list(imgpaths.index)

    for idx in tqdm(image_ids,total=len(image_ids)):
        filepath=imgpaths.loc[idx]["path"]
        path=os.path.join(BASE,"images",filepath)
        try:
            processedImage=processImage(path,filepath)
        except:
            print("skipping ... {}".format(idx))
            continue
        
        pixels = np.array(processedImage).flatten()        # get the pixels as a flattened sequence
        black_thresh = 1
        nblack = 0
        for pixel in pixels:
            if pixel < black_thresh:
                nblack += 1
        n = len(pixels)

        if (nblack / float(n)) > 0.90:
            print("mostly black {}".format(idx))
            continue
            
        # plt.imshow(processedImage)
        joblib.dump(processedImage,os.path.join(SAVEPATH,f"p1_{idx}.pkl"))
        # break








