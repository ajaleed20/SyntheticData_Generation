import tkinter as tk
from datetime import datetime, date, time, timedelta
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import statistics
import sklearn
import math
from sklearn.utils import shuffle
import os
import ctypes



class App(tk.Frame):
    def __init__(self,master=None,**kw):
        #Create a blank dictionary
        self.var = {}
        self.answers = {}
        tk.Frame.__init__(self,master=master,**kw)

        tk.Label(self,text="Indicator Benchmark Value").grid(row=0,column=0)
        self.question1 = tk.Entry(self)
        self.question1.grid(row=0,column=1)

        tk.Label(self,text="Indicator Good Value ( <= Deviation in +/- %)").grid(row=1,column=0)
        self.question2 = tk.Entry(self)
        self.question2.grid(row=1,column=1)

        tk.Label(self, text="Low Limit 1 ").grid(row=2, column=0)
        self.question3 = tk.Entry(self)
        self.question3.grid(row=2, column=1)

        tk.Label(self, text="Low Limit 2 ").grid(row=3, column=0)
        self.question4 = tk.Entry(self)
        self.question4.grid(row=3, column=1)

        tk.Label(self, text="Granularity ").grid(row=4, column=0)
        self.question5 = tk.Entry(self)
        self.question5.grid(row=4, column=1)

        tk.Label(self, text="Start Date (format: yyyy-mm-dd: 2020-12-31)").grid(row=5, column=0)
        self.question6 = tk.Entry(self)
        self.question6.grid(row=5, column=1)

        tk.Label(self, text="End Date (format: yyyy-mm-dd: 2020-12-31)").grid(row=6, column=0)
        self.question7 = tk.Entry(self)
        self.question7.grid(row=6, column=1)

        tk.Label(self, text="Enter Filename to save features as .CSV ").grid(row=7, column=0)
        self.question8 = tk.Entry(self)
        self.question8.grid(row=7, column=1)

        tk.Button(self,text="Go",command = self.collectAnswers).grid(row=8,column=1)


    def collectAnswers(self):
        self.answers['IndicatorBenchmarkValue'] = self.question1.get()
        self.answers['IndicatorGoodValue'] = self.question2.get()
        self.answers['LowLimit1Value'] = self.question3.get()
        self.answers['LowLimit2Value'] = self.question4.get()
        self.answers['Granularity'] = self.question5.get()
        self.answers['SDate'] = self.question6.get()
        self.answers['EDate'] = self.question7.get()
        self.answers['FileName'] = self.question8.get()
        printAnswers(self.answers)
        ROOT_DIR = os.path.abspath(os.curdir)
        time.sleep(10)
        ctypes.windll.user32.MessageBoxW(0, "Synthetic Random Data Generation Done!\n\n File Created at location:\n"+ str(ROOT_DIR), "Synthetic Data Generation", 1)

        exit(0)

#
def printAnswers(answers):
    print("Indicator Benchmark Value: ", answers['IndicatorBenchmarkValue'])
    print("Indicator Good Value: ", answers['IndicatorGoodValue'])
    print("Low Limit1 Value: ", answers['LowLimit1Value'])
    print("Low Limit2 Value: ", answers['LowLimit2Value'])
    print("Granularity: ", answers['Granularity'])
    print("Start Date: ", answers['SDate'])
    print("End Date: ", answers['EDate'])
    print("FileName: ", answers['FileName']+'.csv')
    MaxVal_bm = float(answers['IndicatorBenchmarkValue']) + (
                 (float(answers['IndicatorGoodValue']) / 100) * float(answers['IndicatorBenchmarkValue']))
    MinVal_bm = float(answers['IndicatorBenchmarkValue']) - (
                 (float(answers['IndicatorGoodValue']) / 100) * float(answers['IndicatorBenchmarkValue']))

    print(MinVal_bm)
    print(MaxVal_bm)
    SyntheticValuesCount = 0

    sdate = datetime.strptime(answers['SDate'], "%Y-%m-%d")
    edate = datetime.strptime(answers['EDate'], "%Y-%m-%d")
    deltadate = sdate - edate
    deltadate = abs(deltadate)
    print(deltadate.days)

    if answers['Granularity'] == 'Day':
        SyntheticValuesCount = deltadate.days

    elif answers['Granularity'] == 'Hour':
        SyntheticValuesCount = (deltadate.days) * 24

    elif answers['Granularity'] == 'Minute':
        SyntheticValuesCount = (deltadate.days) * 24 * 60

    elif answers['Granularity'] == 'Second':
        SyntheticValuesCount = (deltadate.days) * 24 * 60 * 60

    elif answers['Granularity'] == 'Tenmin':
        SyntheticValuesCount = (deltadate.days) * 24 * 6

    elif answers['Granularity'] == 'ThirtySec':
        SyntheticValuesCount = (deltadate.days) * ((24 * 60) * 2)

    elif answers['Granularity'] == 'Week':
        SyntheticValuesCount = (deltadate.days) / 7

    else: int(deltadate)

    benchmarkrange = math.ceil(float(0.50 * SyntheticValuesCount))  # 60% values from within benchmark defined
    limitrange     = math.ceil(int(0.30 * SyntheticValuesCount))   # 35% values from within low limit 1 & 2 defined
    outliersrange1  = int(0.05 * SyntheticValuesCount)   # 5% values as outliers or alarm values.
    outliersrange2 =  int(0.05 * SyntheticValuesCount)  # 5% values as outliers or alarm values.
    replicate_range = int(0.10 * SyntheticValuesCount)

    Generated_bm_Values = np.random.uniform(MinVal_bm, MaxVal_bm, benchmarkrange)
    #Generated_bm_Values_dup = [np.random.randrange(MinVal_bm, MaxVal_bm) for i in range(benchmarkrange)]
        #np.random.choice(MinVal_bm, MaxVal_bm, benchmarkrange, replace=True)


    Generated_lm_Values = np.random.uniform(float(answers['LowLimit2Value']), float(answers['LowLimit1Value']), limitrange)
    Generated_out1_Values = np.random.uniform(float(MaxVal_bm+1),float(MaxVal_bm+1.5) , outliersrange1)
    Generated_out2_Values = np.random.uniform(float(float(answers['LowLimit2Value'])-0.5), float(float(answers['LowLimit2Value']) - 1.5), outliersrange2)


    #np.concatenate((a, b), axis=0)

    SyntheticData = np.concatenate((Generated_bm_Values, Generated_lm_Values, Generated_out1_Values, Generated_out2_Values), axis=None)
    df = pd.DataFrame(data=SyntheticData, columns=["Feature"])

    if SyntheticValuesCount == benchmarkrange + limitrange + outliersrange1 + outliersrange2:
        SyntheticData = np.concatenate((Generated_bm_Values, Generated_lm_Values, Generated_out1_Values, Generated_out2_Values), axis=None)

    elif SyntheticValuesCount < (benchmarkrange + limitrange + outliersrange1 + outliersrange2):
        df.drop(df.tail(SyntheticValuesCount - (benchmarkrange + limitrange + outliersrange1 + outliersrange2)).index,inplace=True)

    else:
        for i in range(abs((benchmarkrange + limitrange + outliersrange1 + outliersrange2 + replicate_range) - SyntheticValuesCount)):
            #df = df.append({'Feature': np.random.uniform(MinVal_bm, MaxVal_bm, 1) }, ignore_index=True)
            df = df.append(pd.DataFrame(np.random.uniform(MinVal_bm, MaxVal_bm, 1), columns=['Feature']),ignore_index=True)

    df_replicate = df.head(replicate_range)
    df = df.append(df_replicate, ignore_index=True)
    print("original dataframe:\n")
    print(df)
    df = shuffle(df)
    print("shuffled data frame:\n")
    print(df)
    df.to_csv(answers['FileName'] + '.csv', index=False)


if __name__ == '__main__':
    root = tk.Tk()
    App(root).grid()
    root.mainloop()