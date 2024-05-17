import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model 
from sklearn.metrics import mean_squared_error,r2_score

x,y=datasets.load_diabetes(return_X_y=True)
x=x[:,np.newaxis,2]
x_train,x_test=x[:-20],x[-20:]
y_train,y_test=y[:-20],y[-20:]

model=linear_model.LinearRegression()
model.fit(x_train,y_train)

y_pre=model.predict(x_test)

print(f"""  
coef_:{model.coef_}
mean_squared_error:{mean_squared_error(y_test,y_pre)}
r2_score:{r2_score(y_test,y_pre)}
""")