import os

train_path = ''
test_path = ''

train_file_names = [os.path.splitext(file)[0] for file in os.listdir(train_path) if file.endswith('.wav')]

# generate training.txt
with open('training.txt', 'w') as train_file:
    train_file.write('\n'.join(train_file_names))

test_file_names = [os.path.splitext(file)[0] for file in os.listdir(test_path) if file.endswith('.wav')]

# generate test.txt
with open('test.txt', 'w') as test_file:
    test_file.write('\n'.join(test_file_names))
