import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('./diabetes_temp.csv')    
    pd.set_option('display.max_columns', None)

    print(df.head())

    # drop null and duplicate rows
    dfc = df.dropna().drop_duplicates()
    print(dfc.shape)

    # plotting histogram of Age
    dfc['Age'].plot(color='teal', kind='hist', bins=20)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    # plotting histogram of Age by Outcome
    dfc.hist(column='Age', by='Outcome')
    plt.show()

    # plotting histograms of two columns on the same plot
    df2 = dfc[['Age', 'Glucose']]
    df2.plot(kind='hist',
             alpha=0.7,
             bins=30,
             title='Histogram of Age, Glucose',
             rot=45,
             grid=True,
             figsize=(12,8),
             fontsize=15,
             color=['#ff0000', '#00ff00'])
    plt.xlabel('Age/Glucose')
    plt.ylabel('Count')
    plt.show()

    # Using seaborn
    sns.countplot(x='Outcome', data=dfc)
    plt.show()

    # plotting pie chart
    sizes = dfc['Pregnancies'].value_counts().values[:9]
    labels = dfc['Pregnancies'].value_counts().index[:9]
    colors = ['green','pink','yellow','purple','grey','blue','plum','orange','red']
    plt.pie(sizes,data=dfc,labels=labels,colors=colors,radius=1.25)
    plt.show()

    # plot histograms for all columns
    df2=(dfc.columns[:-1]) #df2 is a list of all columns except the last
    dfc.hist(df2,bins=50, figsize=(20,15), color='lime')
    plt.show()

    # plotting heatmap for correlations between columns
    sns.heatmap(dfc[df2].corr(), annot=True,)
    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))