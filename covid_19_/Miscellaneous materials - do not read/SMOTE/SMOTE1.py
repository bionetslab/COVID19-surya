import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data = pd.read_csv('sales_data.csv')
print(data.head())

# Showing the class imbalance between buyers and non-buyers
data.pivot_table(index='buy', aggfunc='size').plot(kind='bar')

# Create a stratified train/test split. 
# Test set will be 30% of the data.
# Class distribution will be equal for train test and original data

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3, stratify=data.buy)

train.pivot_table(index='buy', aggfunc='size').plot(kind='bar', title='Verify that class distributuion in train is same as input data')

test.pivot_table(index='buy', aggfunc='size').plot(kind='bar', title='Verify that class distributuion in test is same as input data')


##################### SMOTE:
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(train[['time_on_page',	'pages_viewed',	'interest_ski',	'interest_climb']], train['buy'])

# Let’s verify what this has done to our class imbalance. The following code will result in the same bar graph that we created earlier:
pd.Series(y_resampled).value_counts().plot(kind='bar', title='Class distribution after appying SMOTE', xlabel='buy')


# Let’s now redo the model to investigate the effect of SMOTE on our classification metrics. You can redo the model with the following code:
# Instantiate the new Logistic Regression
log_reg_2 = LogisticRegression()

# Fit the model with the data that has been resampled with SMOTE
log_reg_2.fit(X_resampled, y_resampled)

# Predict on the test set (not resampled to obtain honest evaluation)
preds2 = log_reg_2.predict(test[['time_on_page', 'pages_viewed',	'interest_ski',	'interest_climb']])


# We now redo the metrics that we also did in the previous model. This will allow us to compare the two and estimate what the impact of SMOTE has been. You can obtain the confusion matrix as follows:
tn, fp, fn, tp = confusion_matrix(test['buy'], preds2).ravel()
print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue positives: ', tp)

# Let’s also generate the classification report. This can be done as follows:
print(classification_report(test['buy'], preds2)) 
