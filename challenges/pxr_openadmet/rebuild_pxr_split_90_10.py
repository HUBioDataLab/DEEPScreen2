from data_processing import create_regression_training_dataset_from_csv

if __name__ == '__main__':
    metadata = create_regression_training_dataset_from_csv(
        input_csv='challenges/pxr_openadmet/pxr-challenge_TRAIN.csv',
        target_id='pxr_openadmet',
        target_prediction_dataset_path='training_files/target_training_datasets',
        max_cores=10,
        scaffold=True,
        augmentation_angle=10,
        split_ratios=(0.9, 0.1, 0.0),
        seed=42,
    )
    print(metadata)
