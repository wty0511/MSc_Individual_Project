import pandas as pd

def post_processing(df):
    '''Post processing of a prediction file by removing all events that have shorter duration
    than 200 ms.
    
    Parameters
    ----------
    val_path: path to validation set folder containing subfolders with wav audio files and csv annotations
    evaluation_file: .csv file of predictions to be processed
    new_evaluation_file: .csv file to be saved with predictions after post processing
    n_shots: number of available shots
    '''
    new_df = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime'])
    for i, row in df.iterrows():
        if row['Endtime'] - row['Starttime'] > 0.200:
            new_df = new_df.append(row)
        
    return df 
    