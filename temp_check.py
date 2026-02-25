with open('driver_dataset.csv') as f:
    lines = f.readlines()
    print('Total lines:', len(lines))
    print('First 3 lines:')
    for i in range(min(3, len(lines))):
        print(repr(lines[i]))