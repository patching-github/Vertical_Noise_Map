import pandas as pd
import numpy as np

Overall_Data = True
Individual_Data = False
num_samples = 5

if Individual_Data:

    for sample in range(0,num_samples):

        Studio6_Data = 'Comb_PAAE_Analysis_'+str(sample)+'.xlsx'
        H3VR_Data = 'Comb_H3VR_Dir_'+str(sample)+'.xlsx'
        Position_Data = 'Position_Data_0_'+str(sample)+'.xlsx'
        Combined_Data = 'Combined_Data_'+str(sample)+'.xlsx'
        Sheet_Name = 'Combined_Data'

        df_S6 = pd.read_excel(Studio6_Data)

        df_H3 = pd.read_excel(H3VR_Data)

        df_Pos = pd.read_excel(Position_Data)

        df_Comb = pd.merge_asof(df_S6, df_H3, on = 'Time',
                                 allow_exact_matches = True,
                                direction = 'nearest')

        df_Comb = pd.merge_asof(df_Comb, df_Pos, on = 'Time',
                                allow_exact_matches = True,
                                direction = 'nearest')

        df_Comb.to_excel(Combined_Data, Sheet_Name)
        print(f'Complete sample {sample}')

if Overall_Data:
    Studio6_Data = 'Comb_PAAE_Analysis.xlsx'
    H3VR_Data = 'Comb_H3VR_Dir.xlsx'
    Position_Data = 'Position_Data_0_0.xlsx'
    Combined_Data = 'Overall_Data.xlsx'
    Sheet_Name = 'Combined_Data'
    df_S6 = pd.read_excel(Studio6_Data)
    

    df_H3 = pd.read_excel(H3VR_Data)
    df_H3['Time'] = np.array(df_H3['Time'].values, dtype=float)

    df_Pos = pd.read_excel(Position_Data)

    df_Comb = pd.merge_asof(df_S6, df_H3, on = 'Time',
                             allow_exact_matches = True,
                            direction = 'nearest')

    df_Comb = pd.merge_asof(df_Comb, df_Pos, on = 'Time',
                            allow_exact_matches = True,
                            direction = 'nearest')

    df_Comb.to_excel(Combined_Data, Sheet_Name)
    print(f'Complete Overall Data')

##header = []

##for i in df_Comb.head:
##    header.append(str(i))
##
##print(header)
