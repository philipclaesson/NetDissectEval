import pandas as pd
import shutil
import os

def dataSplitter():
        import shutil
        path = "./broden1_224/images/"
        for q in ['q1', 'q2', 'q3', 'q4']: 
            outpath = "./broden1_224_" + q + "/images/"
            index = pd.read_csv("index_" + q + ".csv")
            for folder in ['ade20k', 'dtd', 'opensurfaces', 'pascal']:
                if not os.path.exists(outpath + folder):
                    os.makedirs(outpath + folder)
            for i, row in index.iterrows():
                shutil.copyfile(path + row['image'], outpath + row['image'])
                shutil.copyfile(path + row["object"], outpath + row["object"])
dataSplitter()                                                                                
