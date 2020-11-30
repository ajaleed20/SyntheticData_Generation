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

from matplotlib import pyplot
from pandas import read_csv
from sklearn.utils import shuffle
import os
import ctypes
from util.enums import GanualityLevel



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

        tk.Label(self, text="Granularity(w,d,h,m,s,tm:TenMinute,ts:ThirtySec): ").grid(row=4, column=0)
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

        try:
            tk.Button(self,text="Go",command = self.collectAnswers).grid(row=8,column=1)
        except ValueError as e:
            ctypes.windll.user32.MessageBoxW(0, str(e)+"\n\n Rerun Code to enter values correctly", "ERROR MESSAGE", 1)
            exit(0)


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
        ctypes.windll.user32.MessageBoxW(0,"Generating Synthetic TimeSeries Data......\n")
        time.sleep(5)
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

    try:
        sdate = datetime.strptime(answers['SDate'], "%Y-%m-%d")
        edate = datetime.strptime(answers['EDate'], "%Y-%m-%d")
        deltadate = sdate - edate
        deltadate = abs(deltadate)
        print(deltadate.days)
    except ValueError as e:
        ctypes.windll.user32.MessageBoxW(0,str(e),"ERROR MESSAGE", 1)
        print(e)
        exit(0)
    #"Date Format entered wrong: Rerun Code with following format\n\n 'yy-mm-dddd'"


    if answers['Granularity'] == 'd':
        SyntheticValuesCount = deltadate.days
        level = GanualityLevel.one_day.value

    elif answers['Granularity'] == 'h':
        SyntheticValuesCount = (deltadate.days) * 24
        level = GanualityLevel.one_hour.value

    elif answers['Granularity'] == 'm':
        SyntheticValuesCount = (deltadate.days) * 24 * 60
        level = GanualityLevel.one_minute.value

    elif answers['Granularity'] == 's':
        SyntheticValuesCount = (deltadate.days) * 24 * 60 * 60
        level = GanualityLevel.one_sec.value

    elif answers['Granularity'] == 'tm':
        SyntheticValuesCount = (deltadate.days) * 24 * 6
        level = GanualityLevel.ten_min.value

    elif answers['Granularity'] == 'ts':
        SyntheticValuesCount = (deltadate.days) * ((24 * 60) * 2)
        level = GanualityLevel.thirty_sec.value

    elif answers['Granularity'] == 'w':
        SyntheticValuesCount = (deltadate.days) / 7
        level = GanualityLevel.one_week.value

    else:
        print("Granularity entered wrong: Rerun Code with following entry:\n\n"+
                  "h:hourly\n"+
                  "s: Second\n"+
                  "d: daily\n"+
                  "w:weekly\n"+
                  "ts:thirty seconds\n"+
                  "tm:ten minutes\n")
        ctypes.windll.user32.MessageBoxW(0, "Granularity entered wrong: h:hourly\n s:second\n d:daily\n w:weekly\n ts:thirty seconds\n tm:ten minutes\n", "ERROR MESSAGE", 1)
        exit(0)


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
 #   print("original dataframe:\n")
    print(df)
    df = shuffle(df)
    print("shuffled data frame:\n")
    print(df)


    #df_repeat = df.sample(n=len(df), replace=True)

    #print(df_repeat)
    #print(SyntheticData)
    # print([item for item, count in collections.Counter(df).items() if count > 1])
    # print(df.values)
    # df.to_csv(answers['FileName']+'.csv' , index= False)
    # print([item for item, count in collections.Counter(df).items() if count > 1])

    df['ts'] = pd.DataFrame({'ts': pd.date_range(start=sdate, end=edate, freq=get_freq_by_level(level))})
    df = df.sort_values(by='ts')
    df.to_csv(answers['FileName'] + '.csv', index=False)
    print(df)
    #
    # series = read_csv(answers['FileName'] + '.csv', header=0, index_col=0)
    # print(series.shape)
    # pyplot.plot(series)
    # pyplot.show()
    # print("end")

    #series = read_csv('monthly-car-sales.csv', header=0, index_col=0)


def get_freq_by_level(ganuality_level_value):
    if GanualityLevel.one_hour.value[0] == ganuality_level_value[0]:
        return '60T'
    elif GanualityLevel.three_hour.value[0] == ganuality_level_value[0]:
        return '180T'
    elif GanualityLevel.one_day.value[0] == ganuality_level_value[0]:
        return '1440T'
    elif GanualityLevel.one_week.value[0] == ganuality_level_value[0]:
        return '10080T'
    elif GanualityLevel.ten_min.value[0] == ganuality_level_value[0]:
        return '10T'
    elif GanualityLevel.one_minute.value[0] == ganuality_level_value[0]:
        return '1T'
    elif GanualityLevel.one_sec.value[0] == ganuality_level_value[0]:
        return '0.016666666667T'
    elif GanualityLevel.thirty_sec.value[0] == ganuality_level_value[0]:
        return '0.5T'





def generatedata(answers):
    CountOfRandValues = 100
    OutliersCount = int(CountOfRandValues / 10)
    MaxFeature = answers['IndicatorBenchmarkValue'] + (
                (answers['IndicatorGoodValue'] / 100) * answers['IndicatorBenchmarkValue'])
    MinFeature = answers['IndicatorBenchmarkValue'] - (
                (answers['IndicatorGoodValue'] / 100) * answers['IndicatorBenchmarkValue'])
    print(MinFeature)
    print(MaxFeature)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Synthetic Data Generation")
    windowWidth = root.winfo_reqwidth()
    windowHeight = root.winfo_reqheight()
    print("Width", windowWidth, "Height", windowHeight)

    # Gets both half the screen width/height and window width/height
    positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2)
    positionDown = int(root.winfo_screenheight() / 2 - windowHeight / 2)

    # Positions the window in the center of the page.
    root.geometry("+{}+{}".format(positionRight, positionDown))
    #root.geometry("".format(positionRight, positionDown))
    App(root).grid()
    root.mainloop()