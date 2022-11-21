import preprocessing_utils as preprocess
import pandas as pd

def apply_preprocess(text: str):
    text = text.lower()
    text = preprocess.replace_urls(text)
    text = preprocess.replace_username(text)
    text = preprocess.remove_punctuation(text)
    text = preprocess.tokenize(text)
    return ' '.join(text)

if __name__ == "__main__":
    df_english_train = pd.read_csv("Dataset\H3_Multiclass_Hate_Speech_Detection_train.csv", header = None, skiprows=1)
    df_english_test = pd.read_csv("Dataset\H3_Multiclass_Hate_Speech_Detection_test.csv", header = None, skiprows=1)

    for index in df_english_train.index:
        df_english_train[1][index] = apply_preprocess(df_english_train[1][index])

    for index in df_english_test.index:
        df_english_test[0][index] = apply_preprocess(df_english_test[0][index])

    df_english_train.rename(columns = {0:'label', 1:'tweet',2:'id'}, inplace = True)
    df_english_test.rename(columns = {0:'tweet',1:'id'}, inplace = True)

    df_english_train.to_csv("Processed_Dataset\H3_Multiclass_Hate_Speech_Detection_train_preprocessed.csv", index=False)
    df_english_test.to_csv("Processed_Dataset\H3_Multiclass_Hate_Speech_Detection_test_preprocessed.csv", index=False)