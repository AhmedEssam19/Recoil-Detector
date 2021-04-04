import pandas as pd

from predict import predict
from train import run_training
from utils import transform, get_data_set, create_folds

config = {
    'batch_size': 64,
    'input_size': 224,
    'clf_epochs': 5,
    'reg_epochs': 25,
    'workers': 4,
    'lr': 1e-4,
    'seed': 42,
    'patience': 5
}

sub_df_path = "track1_predictions_example.csv"


def main():
    df = get_data_set('idao_dataset/train', save_to_csv=False)
    create_folds(df, 5, config)

    run_training(1, config, mode='clf')
    run_training(1, config, mode='reg')

    clf_preds, reg_preds = predict(config)

    sub_df = pd.read_csv(sub_df_path)
    sub_df['classification_predictions'] = clf_preds
    sub_df['regression_predictions'] = reg_preds
    sub_df['regression_predictions'] = sub_df['regression_predictions'].apply(transform)

    sub_df.to_csv('Final_Submission.csv', index=False)


if __name__ == '__main__':
    main()
