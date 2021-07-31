import numpy as np
import pandas as pd

#function to get CSV output of our predictions 
def get_csv(model, X_test, y_pred_class):
    name = str(model).partition('(')
    y_pred = pd.Series(y_pred_class, name='predictions')
    measure = pd.Series(X_test['depressed'], name='Measure')
    measure = pd.DataFrame(measure)
    output_data = measure.join(y_pred)
    print(output_data)
    csv = pd.DataFrame(output_data)
    file_name = '~/github/Suicide-Prediction/Suicide-Prediction/output/' + name +'.csv'
    csv.to_csv(file_name, header=True)