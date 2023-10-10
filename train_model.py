import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

#Split arrays or matrices into random train and test subsets.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#Train model with train dataset
rf_model =   ()
rf_model.fit(x_train,y_train)
#Test model with test dataset and evaluate score
y_predict = rf_model.predict(x_test)
score = accuracy_score(y_predict,y_test)
print(f"Obtained model score: {score*100}%")

#save model in model.p
with open('model.p', 'wb') as file:
    pickle.dump({'model':rf_model}, file)
 