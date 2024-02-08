import sys
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('./diabetes_temp.csv')
    pd.set_option('display.max_columns', None)

    # Information about Data/DataFrame
    print(df.shape)

    print(df.head())
    print(df.tail())

    print(df.info())
    print(f"{df.dtypes}\n")
    
    print(f"Null Values:\n{df.isna().sum()}")

    dfn = df.dropna()
    print(dfn.shape)
    df_clean = dfn.drop_duplicates()
    print(df_clean.shape)

    column_headers = list(df.columns)
    print(f"Column headings: {column_headers}")

    data_columns = column_headers[:-1]
    print(data_columns)

    data_corr = df_clean[data_columns].corr()
    print(data_corr)

    corr_data = df_clean.corr().abs()
    cbmi = corr_data['BMI'].sort_values(ascending=False)
    print(f"\n----------sorted correlations----------\n{cbmi}")

    print(df_clean.describe())
    print(df_clean.nunique())

    print(df_clean['Pregnancies'].unique())

    #combine columns of one dataframe to another
    dfnew = pd.concat([df['BMI'], df['Glucose'], df['Outcome']], axis=1)
    print(dfnew.head())

    #filter rows
    df_filter = dfnew[dfnew['Glucose'] <= 180]
    print(f"{df_filter.shape}\n{df_filter.head()}")

    #access rows
    d1 = dfnew.iloc[25,1] # row 25, glucose level
    print(d1)

    #select rows 10-13 in a new dataframe
    d2 = dfnew.iloc[10:14,:]
    print(d2)

    #select rows 10-13 in a new dataframe and reset index
    d3 = dfnew.iloc[10:14,:].reset_index()
    print(d3)

    #select rows 10-13 in a new dataframe and reset index
    #This time, without keeping old indices
    d4 = dfnew.iloc[10:14,:].reset_index(drop=True)
    print(d4)



if __name__ == "__main__":
    sys.exit(int(main() or 0))