import os

# File to snell code in extraction phase

#define function to get image number 
def num(image) :  
    val = 0

    for i in range(len(image)) :
        if image[i] == '(' :
            while True :
                i += 1
                if image[i] == ')' :
                    break
                val = (val*10) + int(image[i])
            break
    
    return val


def getCountOfImage(data_folder):
    max_ben = 0
    max_mal = 0
    classes = ['benign', 'malignant']
    label = 0
    labels = []
    benign = 0
    malignant = 0
    for cname in os.listdir(data_folder):
       for filename in sorted (os.listdir(os.path.join(data_folder,cname))):
           if not '_mask' in filename :
               if 'benign' in filename :
                 if num(filename) > max_ben:
                   max_ben = num(filename)
                 benign +=1
               elif 'malignant' in filename:
                 if num(filename) > max_mal:
                   max_mal = num(filename)
                 malignant +=1 
           
    return int(benign), int(malignant), max_ben, max_mal