Author: Zhicong Sun<br />Data: 2021.5.30
<a name="RXo4j"></a>
## Abstrct
This my the mini project of CSE5002 Intelligent Data Analysis of South University of Science and Technology.
<a name="71418df4"></a>
## 1. Introduction
This my the mini project of CSE5002 Intelligent Data Analysis. The first section of this report describes the problems to be solved and the work done.  The second section introduces the methods and materials used, mainly the process of data analysis and preprocessing.  The third section introduces two experiments in detail. Experiment 1 uses the attribute dataset and compares five classification models. Experiment 2 uses the attribute dataset and the adjacency list dataset to trains and test on the final ten-layer neural network classifier.  After adding the adjacency list data set, the accuracy of the classifier is significantly improved by 50%, the final accuracy on the test dataset reached 80%.
<a name="2a0869bf"></a>
### 1.1 What is the problem to solve
In this mini project, an attributed social network at MIT (MIT, for short) is used as a toy example.<br />The original dataset comes from [1]. To simulate the above scenario, the related term to “age” is<br />“class year” in MIT dataset. Therefore, we adopt “class year” as the label in our mini project. We<br />have preprocessed MIT dataset by removing the lines with 0 presented in “class year”, which<br />finally yields 5298 rows of data.<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622355932884-eb734b8e-1e83-45ec-8310-31513bb7cce2.png#height=207&id=u66db3f2e&name=image.png&originHeight=414&originWidth=1027&originalType=binary&size=359916&status=done&style=none&width=514)<br />Assume that there are some missing labels of “class year”. We need to predict the missing labels<br />(a multi-class classification problem) based on two sources of information. One comes from node<br />attributes, while another is from network topology. Specifically, our dataset consists of<br />

- attr.csv: node_id,degree,gender,major,second_major,dormitory,high_school (5298 rows)
- adjlist.csv: node_id,neighbor_id_1,neighbor_id_2,… (5298 rows)
- label_train.csv: node_id,class_year (4000 rows)
- label_test.csv: node_id,class_year (1298 rows)


<br />where node_id (each corresponds to a person) ranges from 0 to 5297. In this mini project, our<br />training set contains node_id from 0 to 3999, and testing set contains node_id from 4000 to 5297.<br />
<br />The **objective** is to train a classifier, utilizing node attributes, or network topology, or both, to<br />make good predictions for the missing labels in testing set.<br />
<br />[1] Traud, Amanda L., et al. "Comparing community structure to characteristics in online collegiate<br />social networks." SIAM review 53.3 (2011): 526-543
<a name="87ee1ef1"></a>
### 1.2 What has this project done
The task I have done in this project can be divided into five parts:

1. Data analysis
2. Data preprocessing
3. Model evaluation and selection
4. Comparing the performance between one source of information and two sources of information
5. Using the best classification to predict the missing label
<a name="vL2yj"></a>
### 1.3 How to set up the environment
Platform: Macbook Air<br />System: macOS Big Sur 11.2.1<br />Main running environment of this project:

- Anaconda 2020.11
- Spyder 4.1.5
- python 3.8.5
- pytorch 1.8.1
- scikit-learn 0.23.2
- numpy 1.19.2
- pandas 1.1.3
- matplotlib 3.3.2
<a name="GiXiU"></a>
### 1.4 How to use this source code

- **data_analysis.ipynb** is used for data analysis. 
- **clf_without_adjlist.ipynb** corresponds to the content of Experiment 1, using the attribute data set to cross-validate and compare five models, and finally use the integrated model to train and predict on the test set. 
- **clf_with_attr.ipynb** corresponds to the Experiment 2, using the attribute dataset and the adjacency list dataset, the classifier is a neural network with ten layers, the code first cross-validates the training data, and then all training data is trained and performed on the test set  prediction.  
- The above 3 files need to be opened in Jpydter and then run completely and respectively. It is best not to run only one of the fragments.
<a name="oe5cn"></a>
## 2 Methods and Materials 
<a name="AdccI"></a>
### 2.1 Data analysis
<a name="g2bUE"></a>
#### 2.1.1 First meeting with data

1. **First meeting with attribute data and training labels: **

![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622347025283-683c1e02-5a96-4bcb-95da-ec4c3fb38a96.png#height=233&id=u6e257032&margin=%5Bobject%20Object%5D&name=image.png&originHeight=310&originWidth=280&originalType=binary&size=20695&status=done&style=none&width=210)<br />It can be seem that the attribute data set has 5298 row and 6 column, which means there are 6 features. Because the data set has no feature names, so we use feature i to both training dataset and testing dataset.<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622347497770-fa381b88-6234-4082-ad9d-6beacf179e92.png#height=22&id=uea5295f0&name=image.png&originHeight=29&originWidth=749&originalType=binary&size=5216&status=done&style=none&width=562)<br />The format of the changed training dataset is:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622348263856-e50af594-6bb5-48ee-af9d-4942b15f7df3.png#height=112&id=u86a417f2&name=image.png&originHeight=149&originWidth=503&originalType=binary&size=12869&status=done&style=none&width=377)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622348491459-c2d4d119-78bb-4593-9d99-183732165a2d.png#height=111&id=u955e4bb8&name=image.png&originHeight=148&originWidth=267&originalType=binary&size=6830&status=done&style=none&width=200)<br />The format of the changed testing dataset is:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622354849818-af6680be-00e8-49fd-9e33-09ee03b0627f.png#height=110&id=u4de11470&name=image.png&originHeight=147&originWidth=511&originalType=binary&size=13153&status=done&style=none&width=383)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622354874420-1c930344-c338-4a28-8b65-624f3a0a5ad1.png#height=119&id=uf8f49ee3&name=image.png&originHeight=158&originWidth=257&originalType=binary&size=6530&status=done&style=none&width=193)

2. <br />
<a name="NY3xS"></a>
#### 2.1.2 Data exploration 

1. **Check if the attrbute data has missing value**

There are no missing value, we don't need to supplement any value.<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622356606986-9bcd499a-e39d-42bd-af99-3ac25f130748.png#height=280&id=u06716630&name=image.png&originHeight=373&originWidth=314&originalType=binary&size=24076&status=done&style=none&width=236)

2. **Label visualization**

Label visualization of training dataset:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622358131761-f52d9bc2-acf8-43b1-9ac7-d150af43b71e.png#height=293&id=u44fccbeb&margin=%5Bobject%20Object%5D&name=image.png&originHeight=586&originWidth=879&originalType=binary&size=17803&status=done&style=none&width=440)<br />Label visualization of testing dataset<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622358076261-edfa760f-176a-4c55-a193-c7694c91378c.png#height=292&id=uc0f31c31&name=image.png&originHeight=584&originWidth=885&originalType=binary&size=15652&status=done&style=none&width=443)<br />We counted the number of different labels in training dataset and test dataset, and the total number of different labels in training dataset and testing dataset:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622358570475-ce393bae-f673-489b-b04d-153d576a16d5.png#height=47&id=u621f0966&name=image.png&originHeight=47&originWidth=229&originalType=binary&size=3881&status=done&style=none&width=229)<br />There are 29 labels in the train dataset while 22 labels in test dataset, and the total number of labels is 32.It means that some labels of the test dataset are not available in the training dataset, and some labels of the training set do not appear in the test dataset. So we should encode all 32 labels. <br />

3. **Correlation matrix visualization and statistics for features**

We analyze the correlation of different features in order to study whether it is necessary to combine different features to construct new features. The following results show that the correlation between features is not high：<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622359157896-5ed47814-916c-4ab2-bfef-967203cf2af0.png#height=426&id=u43187d1d&name=image.png&originHeight=568&originWidth=547&originalType=binary&size=29689&status=done&style=none&width=410)

4. **Divide features into categorical value and continous value**


<br />We counted the different values contained in each feature, the results are as follows:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622359361714-83a6bacf-f99b-4456-9d87-4d897a896da8.png#height=204&id=u96718c31&name=image.png&originHeight=272&originWidth=676&originalType=binary&size=35414&status=done&style=none&width=507)<br />Based on the above results, we can divide features into caegorical features and continuous features. feature1 and feature2 can be regarded as categorical feature, feature3, feature4, feature5 and feature6 can be regarded as continous feature:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622359391734-23dafc75-e6d3-40cc-950f-06df904923ca.png#height=51&id=u7fe9292f&name=image.png&originHeight=51&originWidth=575&originalType=binary&size=5571&status=done&style=none&width=575)

5. **Continous features visualization**

![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622359740298-6462036d-cb75-40dd-832c-e42de397d0fe.png#height=192&id=u92ac9558&name=image.png&originHeight=256&originWidth=385&originalType=binary&size=7591&status=done&style=none&width=289)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622359784628-56c6e71e-dc8d-4e03-a936-7fb4ff25f040.png#height=190&id=u18987762&name=image.png&originHeight=253&originWidth=390&originalType=binary&size=7564&status=done&style=none&width=293)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622359808061-be4a36ce-0e9c-4e86-9815-b8e47cdd5d67.png#height=190&id=u3b2edf10&name=image.png&originHeight=253&originWidth=384&originalType=binary&size=8391&status=done&style=none&width=288)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622359828677-d1f49d27-2499-4fed-92ca-9d4527d309fa.png#height=189&id=u7342851a&margin=%5Bobject%20Object%5D&name=image.png&originHeight=252&originWidth=386&originalType=binary&size=9071&status=done&style=none&width=290)<br />By analyzing the above four figures, we can draw a conclusion:

- The continuous features have almost no outliers.
-  The feature3, feature6 seem to obey the normal distribution. 
-  It is hard to figure out if the feature4 and feature5 obey the normal distribution, but I tend to treat them as a feature following the normal distribution. Because it appears to follow a normal distribution if we supplement the data on the left side of the axis (even if the data is meaningless). 



6. **Categorical features visualization**

![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622360036501-f4e5adc0-7497-4aaf-8246-473d3dad097e.png#height=204&id=ufb99bbf3&name=image.png&originHeight=272&originWidth=390&originalType=binary&size=8886&status=done&style=none&width=293)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622360058929-cf5126b0-c4d1-450e-ba35-d091f15e9e07.png#height=204&id=u62b5ae21&name=image.png&originHeight=272&originWidth=389&originalType=binary&size=7684&status=done&style=none&width=292)<br />It is hard to determin which features is ordered. So we regard them as disorderd features and tolerate the loss of information caused by using features as an unordered variable.
<a name="vnNuh"></a>
### 2.2 Data preprocessing
Based on the above data analysis, our data preprocessing can be divided into the following 3 steps：

1. The categorical features in attribute dataset are encoded as dummy variables.
1. Do feature scaling for continuous features, here we mainly focus on standardization.
1. Deal with the adjacency list and convert it to the accessible  matrix.
<a name="P3hic"></a>
#### 2.2.1 Dummy coding of categorical features
After dummy variable processing, the number of attributes increased from 6 to 13:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622362649295-f61ccf9d-a5f4-4295-ba6e-1832a0deb4a8.png#height=68&id=uff7daf53&name=image.png&originHeight=90&originWidth=757&originalType=binary&size=14156&status=done&style=none&width=568)
<a name="zvdh4"></a>
#### 2.2.2 Features scaling of continous features
After standardizing continuous features, the data of attribute is as follows：<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622363058993-5e455883-6c49-4b9b-ba68-06c8b3d02177.png#height=125&id=u602ea90c&name=image.png&originHeight=167&originWidth=958&originalType=binary&size=21425&status=done&style=none&width=719)
<a name="oqP2a"></a>
#### 2.2.3 Process adjacent list
In the original adjacency list, the values placed in each row represent the neighbor samples connected with the current sample. We first convert this list to adjacency matrix, and then to reachability matrix. Every value in the reachability matrix is either zero or one. For example, matrix (i, j) = 1 means that the i-th sample and the j-th sample are connected,otherwise they don't connect. <br />**The purpose** of this processing is to regard whether each sample is connected with any other sample as a feature. The final reachable matrix is a 5298 x 5298 matrix with the following format:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622355857588-5e9479bb-a14e-41d6-837b-2a246dfd41e0.png#height=326&id=hna0A&margin=%5Bobject%20Object%5D&name=image.png&originHeight=434&originWidth=636&originalType=binary&size=20058&status=done&style=none&width=477)<br />Note:

- If the accessible matrix should be used as train data, we need to combine this matrix and other features.
- In this case, we add 5298 features, so the total number of labels reached 5311.
- These new features are used to indicate whether the current sample point is connected with another sample point
<a name="TL8QD"></a>
## 3 Experiments and discussion
<a name="zIZzu"></a>
### 3.1 Experiment 1 
<a name="ou1XI"></a>
#### 3.1.1 Overview: 
Classification using attribute dataset but not adjacent dataset. There are several characteristics in this experiment：

1. **Dataset**

This experiment only use the attibute dataset but not the network topology data to train and predict.

2. **Classifier**

This experiment tests the performance of five models on the dataset，these five models are: K Nearest Neighbor(KNN), Support Vector Machine (SVM), Decision Tree, Neural Netwrok and Ensembel Model.

3. **Procedure**

This experiment first uses k-fold cross validation to compare the performance of different models on the validation set, then uses the whole attrbute training dataset to train the model with the best performance, and finally makes prediction on the testing dataset.

4. **Performance evaluation**

The model performance evaluation criterion used in this experiment is the **Accuracy**. All the more complex evaluation criteria are not used because this is only our preliminary test. Although the accuracy has other shortcomings, it can most intuitively reflect the prediction performance. In Experiment 2, we will use various measures to evaluate the performance of the model.
<a name="v0rjR"></a>
#### 3.1.2 Results and discuss
**Results of k-fold cross validation:**<br />**​**

![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622365001696-3b56ea37-a67d-4f78-a0b5-d37e07699636.png#height=170&id=uefd27ce4&name=image.png&originHeight=170&originWidth=395&originalType=binary&size=17445&status=done&style=none&width=395)

- The testing accuracy in the above figure is the accuracy on the verification set.
- These results show that the performance of neural network is the best among all Non-ensemble models, but the accuracy can only reach 0.3. 


<br />**Rresult of using all attribute training dataset in the Ensemble model:**<br />**​**

![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622391646944-73d66352-a15c-4649-947e-5ba1aaf00ec5.png#clientId=u2dbe16c2-0a53-4&from=paste&height=36&id=p89ot&name=image.png&originHeight=36&originWidth=423&originalType=binary&size=7176&status=done&style=none&taskId=u31432930-0a24-45bb-8e56-92d73037f4b&width=423)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622391684501-e573a755-3c97-4224-a82b-1803d9f8cdc9.png#clientId=u2dbe16c2-0a53-4&from=paste&height=74&id=vwGHj&name=image.png&originHeight=74&originWidth=313&originalType=binary&size=10255&status=done&style=none&taskId=u6afea7cb-3c41-45ff-a9d8-b0774507ab6&width=313)

- The accuracy of the ensembel model reaches 31%,  and the Micro F1-score reaches 31%.
- It shows that only using attribute data to train the model can only improve the accuracy by about 20% compared with random guessing. 
- We  need to make full use of the information provided by the topology. 
<a name="J0Fle"></a>
### 3.2 Experiment 2
<a name="PhgBX"></a>
#### 3.2.1 Overview
Classification using adjacent dataset and attribute dataset. There are several characteristics in this experiment：

1. **Dataset**

This experiment use both the **attibute dataset** and the **adjacency datase**t of network topology to train and predict. The preprocessing of training dataset and test set is the same

2. **Classifier**

This experiment tests the performance of five models on the dataset，these five models are: Neural Netwrok. <br />The network consists of input layer, output layer and 8 hidden layers. The number of neurons in input layer is 5311 (equal to the number of features), and that in output layer is 32 (equal to the total number of classes). The details are as follows:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622374502388-72c6b1f1-7b60-41d7-b3ee-5734d9cc78ff.png#height=161&id=OJv1v&name=image.png&originHeight=215&originWidth=513&originalType=binary&size=33896&status=done&style=none&width=385)<br />In addition, compared with other activation functions, relu has the following advantages: for linear functions, relu is more expressive, especially in deep networks; For nonlinear functions, because the gradient of non negative interval is constant, there is no vanishing gradient problem, which makes the convergence rate of the model maintain a stable state. Here is a little description of what is the gradient vanishing problem: when the gradient is less than 1, the error between the predicted value and the real value will decay once per layer of propagation. If sigmoid is used as the activation function in the deep model, this phenomenon is particularly obvious, which will lead to the stagnation of the model convergence.

3. **Procedure**

This experiment first uses k-fold cross validation to test the performance of  models on the validation set, then uses the whole training dataset to train the model with the best performance, and finally makes prediction on the testing dataset.

4. **Performance evaluation**

The evaluation criteria for classification problems include accuracy, precision, recall, F1 score, etc. we use** **Accuracy, Precision and F1-score to measure the quality of the model. The multi classification method is different from the binary classification method, so we use the Micro F1 score, Micro Precision, Micro Precision and Weighted Precision.  In these criterias, **Micro F1-score **and** Accuracy **are the most important criterias we refer to. <br />There are precision and recall rates for binary classification problems. Each class of muti-classification problems has their own precision rate, recall rate, TP, TN, FP and FN. In this project, P is used to represent the precision rate, R is the recall rate. Micro-method F1-score caculates the average of TP, TN, FP and FN respectively, and then calculate P, R and F 1.<br />The formulas of Micro F1-score are as follows:<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622391242848-02b0f45d-959d-496d-934c-db95b8305500.png#clientId=u2dbe16c2-0a53-4&from=paste&height=149&id=u4b685bca&name=image.png&originHeight=198&originWidth=388&originalType=binary&size=13410&status=done&style=none&taskId=u9f273fae-ba3d-40c7-a68a-703ec7ac9f0&width=291)
<a name="vYXTs"></a>
#### 3.2.2 Results and discuss
![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622390729605-0fe6f629-81e3-45d1-ad05-3b7643924a82.png#clientId=u2dbe16c2-0a53-4&from=paste&height=108&id=ua907f72c&name=image.png&originHeight=108&originWidth=430&originalType=binary&size=11728&status=done&style=none&taskId=ucdd25466-4c9b-4247-b202-106b778a5d8&width=430)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/753325/1622390800910-f6181e5d-309b-4809-ba80-1786ca62d5d2.png#clientId=u2dbe16c2-0a53-4&from=paste&height=75&id=u35eafd5d&name=image.png&originHeight=75&originWidth=343&originalType=binary&size=11075&status=done&style=none&taskId=u83df49bd-4f31-4846-869f-27911060464&width=343)

- The accuracy of the current model reaches 80%,  and the Micro F1-score reaches 80%, which indicates that the connection relationship of samples can provide more information for the model and improve the fitting degree of the model. 
- **Using both two sources ofinformation is better than using a single source of information.**
<a name="KTC4H"></a>
## 4 limitation and future work
This project first visualized and statistically analyzed the dataset, divided the attribute data into categorical features and continuous features, processed the categorical features as dummy variables, and standardized the continuous features.  Then two experiments were carried out.  <br />Experiment 1 uses attribute dataset to train five models and k-fold cross-validation, and the accuracy is used as to evaluate models. The accuracy of the models on the test set can only reach about 0.30. And the results show that the neural network may have better performance.  Experiment 2 used the attribute dataset and the adjacency list dataset which represents the network topology. The adjacency list is processed into a reachable matrix. The connection relationship between each sample and any other sample is a feature.  The neural network has ten layers and the model is trained in batches, using F1-score and accuracy as metrics.  In the end, both the **accuracy and Micro F1-score **of the model on the test set reached **0.80**, which was significantly better than all the models in Experiment 1.<br />The **limitation** of the final model proposed in this project is that the connection relationship between each sample and any other sample is used as a feature, which brings the number of features to 5,311. If it is a larger social network, the number of features will increase linearly with the number of samples. Therefore, the dimensionality reduction processing of this topology data can be a solution, and another solution is to use a graph neural network. 
