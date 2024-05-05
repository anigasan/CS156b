from sklearn.impute import SimpleImputer
import pandas as pd


df = pd.read_csv('/Users/anigasan/Desktop/cs156bstuff/CS156b/train/train copy.csv')

imputer_floats = SimpleImputer(strategy='mean');

df_floats = df.select_dtypes('float64')

df_floats_final = pd.DataFrame(df_floats[['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Pneumonia','Pleural Effusion','Pleural Other','Fracture','Support Devices']])


df_floats_imputed = pd.DataFrame(imputer_floats.fit_transform(df_floats_final), columns = df_floats_final.columns)

df_non_floats = df.select_dtypes(include = ['int64', 'object'])

final_dataframe = pd.concat([df_non_floats, df_floats_imputed], axis=1)

print(final_dataframe.info())
print(df.info())
print(final_dataframe.head())
