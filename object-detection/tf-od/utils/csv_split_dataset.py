import os
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit 

def split_dataset(csv_path, images_dir):
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)

    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    split = splitter.split(df, groups=df['image_name'])
    train_inds, temp_inds = next(split)
    train_data = df.iloc[train_inds]
    temp = df.iloc[temp_inds]

    splitter = GroupShuffleSplit(test_size=.50, n_splits=2, random_state = 7)
    split = splitter.split(temp, groups=temp['image_name'])
    val_inds, test_inds = next(split)
    val_data = temp.iloc[val_inds]
    test_data = temp.iloc[test_inds]

    # Function to copy images to the respective directories
    def copy_images(dataframe, destination_dir):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        for image_name in set(dataframe['image_name']):
            image_path = os.path.join(images_dir, image_name)
            destination_path = os.path.join(destination_dir, image_name)
            os.rename(image_path, destination_path)

    # Copy images to the respective directories for train, validation, and test sets
    parent_dir = "/".join(images_dir.split('/')[:-1])
    copy_images(train_data, f'{parent_dir}/train')
    copy_images(val_data, f'{parent_dir}/val')
    copy_images(test_data, f'{parent_dir}/test')

    # Save the corresponding CSV files for train, validation, and test sets
    train_data.to_csv(f'{parent_dir}/train/faces.csv', index=False)
    val_data.to_csv(f'{parent_dir}/val/faces.csv', index=False)
    test_data.to_csv(f'{parent_dir}/test/faces.csv', index=False)
    
    os.rmdir(images_dir)
    os.remove(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split object detection dataset into training, validation, and testing sets.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file in YOLO format.')
    parser.add_argument('--images', type=str, required=True, help='Path to the directory containing the images.')
    args = parser.parse_args()

    split_dataset(args.csv, args.images)
