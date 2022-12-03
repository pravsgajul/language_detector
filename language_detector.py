import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data=pd.read_csv(r"languages.csv")

x=np.array(data['Text'])
y=np.array(data['Language'])

cv=CountVectorizer()
X=cv.fit_transform(x)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=38)

model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)