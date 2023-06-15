from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data import data_loader

data = data_loader()
X_train, X_test, y_train, y_test = train_test_split(data.drop('price', axis=1), 
                                                    data['price'], 
                                                    test_size=0.2, 
                                                    random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error:", mae)