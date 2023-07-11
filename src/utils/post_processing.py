# This code is modified from DCASE Challenge 2022 https://github.com/c4dm/dcase-few-shot-bioacoustic
import pandas as pd
import os
import csv
# def post_processing(df, cfg, mode):
#     '''Post processing of a prediction file by removing all events that have shorter duration
#     than 200 ms.
    
#     Parameters
#     ----------
#     val_path: path to validation set folder containing subfolders with wav audio files and csv annotations
#     evaluation_file: .csv file of predictions to be processed
#     new_evaluation_file: .csv file to be saved with predictions after post processing
#     n_shots: number of available shots
#     '''
#     new_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime'])
#     for i, row in df.iterrows():
#         if row['Endtime'] - row['Starttime'] > 0.099:
#             new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)

        
#     return new_df 
    
    
    
def post_processing(df, cfg, mode):
    if mode == 'val':
        val_path = cfg.path.val_dir
    if mode == 'test':
        val_path = cfg.path.test_dir
    n_shots=5
    
    dict_duration = {}
    folders = os.listdir(val_path)
    for folder in folders:
        # print(folders)
        files = os.listdir(os.path.join(val_path, folder))
        for file in files:
            if file[-4:] == '.csv':
                audiofile = file[:-4]+'.wav'
                annotation = file
                events = []
                
                with open(os.path.join(val_path, folder,annotation)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        for row in csv_reader:
                            if row[-1] == 'POS' and len(events) < n_shots:
                                events.append(row)
                min_duration = 10000
                for event in events:
                    if float(event[2])-float(event[1]) < min_duration:
                        min_duration = float(event[2])-float(event[1])
                dict_duration[audiofile] = min_duration
    # print(dict_duration)
    new_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime'])
    # print(dict_duration)
    for i, row in df.iterrows():
        if row['Endtime'] - row['Starttime'] >= dict_duration[row['Audiofilename']] * 0.6:
            new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
        # else:
        #     print(row['Audiofilename'])
        #     print(row['Endtime'] - row['Starttime'])
        #     print(dict_duration[row['Audiofilename']])
    return new_df