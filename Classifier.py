import pandas as pd, numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

cov_type=pd.read_csv('covtype.data',header=None)
cov_type=cov_type.as_matrix()
X=(cov_type[:,:cov_type.shape[1]-1]).astype('float64')
y=cov_type[:,cov_type.shape[1]-1]-1

ml=MultiLabelBinarizer()
y_onehot=ml.fit_transform(y.reshape(-1,1))

import pandas as pd, numpy as np

n_classes = y_onehot.shape[1]

missclassif_cost_matrix = np.zeros((n_classes, n_classes))
np.random.seed(1)
for i in range(n_classes - 1):
    for j in range(i + 1, n_classes):
        cost_missclassif = np.random.gamma(1, 5)
        missclassif_cost_matrix[i, j] = cost_missclassif
        missclassif_cost_matrix[j, i] = cost_missclassif


C=np.array([missclassif_cost_matrix[i] for i in y])


C=C*(X[:,9]/2000).reshape(-1,1)
C=C/(1+X[:,22]*20).reshape(-1,1)
C=C/(1+X[:,30]*7).reshape(-1,1)
C=C*(1+X[:,41]*3).reshape(-1,1)
C=C*(1+X[:,47]*8).reshape(-1,1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, C_train, C_test, y_train, y_test = train_test_split(X, C, y, test_size=.5, random_state=1)
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



from costsensitive import WeightedAllPairs, WeightedOneVsRest, RegressionOneVsRest, \
                            FilterTree, CostProportionateClassifier
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

### Keeping track of the results for later
name_algorithm=list()
test_accuracy=list()
test_cost=list()

################ Note ################
### These reduction methods require classifiers supporting sample weights.
### If your favorite classifier doesn't, you can convert it to
### an importance-weighted classifier like this:
ClassifierSupportingSampleWeights = CostProportionateClassifier(LogisticRegression())
### (replace LogistRegression for your classifier)

#### Benchmark : Logistic Regression with no weights
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
name_algorithm.append("Multinomial Loss")
test_accuracy.append(np.mean(preds_lr==y_test))
test_cost.append(C_test[np.arange(C_test.shape[0]), preds_lr].sum())



#### 2. Weighted All-Pairs - simpler cost-weighting schema
costsensitive_WAP2 = WeightedAllPairs(LogisticRegression(solver='lbfgs'), weigh_by_cost_diff=True)
costsensitive_WAP2.fit(X_train, C_train)
preds_WAP2 = costsensitive_WAP2.predict(X_test, method='most-wins')
name_algorithm.append("Weighted All-Pairs (Simple importance weights)")
test_accuracy.append(np.mean(preds_WAP2==y_test))
test_cost.append(C_test[np.arange(C_test.shape[0]), preds_WAP2].sum())

import pandas as pd

results = pd.DataFrame({
    'Method' : name_algorithm,
    'Accuracy' : test_accuracy,
    'Total Cost' : test_cost
})
results=results[['Method', 'Total Cost', 'Accuracy']]
results.set_index('Method')

print(results)
