import os
from ensemble import *
from sklearn.metrics import classification_report

IMG_SIZE = 12 * 12

def get_path(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

def grayscale(src_path,dst_path):
    #imgs = get_path(src_path)
    for src in os.listdir(src_path):
        dst = os.path.join(dst_path,src)
        Image.open(os.path.join(src_path,src)).resize((24,24)).convert('L').save(dst)

def extract(path):
    features = []
    cnt = 0   
    for img in os.listdir(path):
        f = NPDFeature(np.array(Image.open(os.path.join(path,img)))).extract()
        #print(f)
        features.append(f)
        cnt = cnt + 1
    return cnt,features

def init_features():
    grayscale('datasets\\original\\face','datasets\\gray\\face')
    grayscale('datasets\\original\\nonface','datasets\\gray\\nonface')
    (cnt0,features0) = extract('datasets\\gray\\nonface')
    (cnt1,featrues1) = extract('datasets\\gray\\face')
    y = np.ones((cnt0+cnt1,1))
    y[:cnt0] = -1
    x = np.array([features0,featrues1]).reshape((1000,-1))
    AdaBoostClassifier.save(x,'x.ds')
    AdaBoostClassifier.save(y,'y.ds')

if __name__ == "__main__":
    # write your code here
    if not(os.path.isfile('x.ds') and os.path.isfile('y.ds')):
    	init_features()
    x = AdaBoostClassifier.load('x.ds')
    y = AdaBoostClassifier.load('y.ds')
    print('the size of X:',x.shape)
    print('the size of y:',y.shape)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33)

    #AdaBoosting
    AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),10)
    AdaBoost.fit(x_train,y_train)
    print('the wrong number of train sample:',AdaBoost.is_good_enough(x_train,y_train))

    #show the result
    target_names = ['NEGATIVE', 'POSITIVE']
    y_pred = AdaBoost.predict(x_test)
    result = classification_report(y_test,y_pred,target_names=target_names)

    print(result)
    with open("report.txt","w") as f:
        f.write(result)


