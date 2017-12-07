
# coding: utf-8

# # ImageNetで学習済みのモデル(Inception V3)を使って入力画像を予測

# ## クラスの読み込み

# In[1]:


##Kerasや数値計算のためのクラスの読み込み
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

## 画像を表示するクラスの読み込み
import cv2
print(cv2.__version__)
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

## SSL証明書が正しくないサイトに対してPythonでアクセスする
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# # 既存の学習済みモデル(Inception V3)の取り込み & インスタンス化

# In[2]:


model = InceptionV3(weights='imagenet', include_top=True)


# ## 入力画像の選択

# In[40]:


#<banana>
#filename = "./datasets/validation/banana/2675998934_8a74f15c80.jpg"   #テーブルに置かれたバナナ
#filename = "./datasets/validation/banana/banana.jpg"                  #皮をむいたバナナ

#<apple>
#filename = "./datasets/validation/apple/100_5545_mid.jpg"            #地面に落ちたリンゴ
#filename = "./datasets/validation/apple/apple2.jpg"                  #赤いリンゴ

#<car>
#filename = "./datasets/validation/car/car1.jpg"                     #自動車1
#filename = "./datasets/validation/car/car2.jpg"                     #自動車2
#filename = "./datasets/validation/car/car3.jpg"                     #スポーツカー
filename = "./datasets/validation/car/car4.jpg"                     #事故でつぶれた自動車

#<bicycle>
#filename = "./datasets/validation/bicycle/bicycle1.jpg"            #二人乗り自転車
#filename = "./datasets/validation/bicycle/bicycle2.jpg"            #人が自転車に乗った絵
#filename = "./datasets/validation/bicycle/bicycle3.jpg"            #マウンテンバイク
#filename = "./datasets/validation/bicycle/bicycle4.jpg"            #折りたたまれた？自転車


# In[41]:


## 選択した画像の表示（確認用）
img2 = cv2.imread(filename)
img2_rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
plt.imshow(img2_rgb)

## 後続の推論処理へ渡す画像の指定
img = image.load_img(filename, target_size=(299, 299)) #Inception V3の場合は、sizeは299,299。 cf. VGG16の場合は244,244


# # 選択した画像の判別（推論）

# In[42]:


## 入力画像の行列化
x = image.img_to_array(img)

## 4次元テンソル
x = np.expand_dims(x, axis=0)

## 予測
preds = model.predict(preprocess_input(x))  #Inception V3モデルで入力画像が何か推論　

results = decode_predictions(preds, top=5)[0]

## 結果出力(上位5つの候補を確信度とともに表示)
print("選択した画像は下記に該当すると予測されます。※該当すると予測される上位5件を表示。")
print("")
for result in results:
    print(result)


# In[ ]:





# In[ ]:





# In[ ]:




