import os
import random
import shutil
import argparse

def split(train, test):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='path to dataset', default='Mito')
    parser.add_argument('--train_size', default=0.75, help='training set size')
    args = parser.parse_args()

    source = os.path.join(args.src, train)
    dest = os.path.join(args.src, test)

    source_Mito = os.path.join(source, 'target')
    source_TL = os.path.join(source, 'source')

    dest_Mito = os.path.join(dest, 'target')
    dest_TL = os.path.join(dest, 'source')

    files = os.listdir(source_Mito)
    no_of_files = round(len(files) * (1 - args.train_size))

    for file_name in random.sample(files, no_of_files):
        shutil.move(os.path.join(source_Mito, file_name), dest_Mito)
        shutil.move(os.path.join(source_TL, file_name), dest_TL)

def main():
    split('train', 'test')
    split('test', 'val')
    print("finish splitting data")
    
if __name__ == '__main__':
    main()
