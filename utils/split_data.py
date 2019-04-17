import csv
import random


base_dir = './data/sentinel/'
filename = base_dir + 'sentinel_locations.csv'
train_filename = base_dir + 'parcel_segmentation_train_sentinel.csv'
test_filename = base_dir + 'parcel_segmentation_test_sentinel.csv'
val_filename = base_dir + 'parcel_segmentation_val_sentinel.csv'

train_filename_fill = base_dir + 'parcel_segmentation_train_sentinel_fill.csv'
test_filename_fill = base_dir + 'parcel_segmentation_test_sentinel_fill.csv'
val_filename_fill = base_dir + 'parcel_segmentation_val_sentinel_fill.csv'


with open(filename, 'r') as f:
  csv_reader = csv.reader(f)
  ids = set()
  for index, row in enumerate(csv_reader, -1):
    if index == -1:
      continue
    ids.add(int(row[0]))

ids = list(ids)
random.shuffle(ids)
train = int(0.8 * len(ids))
test = int(0.1 * len(ids))
val = len(ids) - train - test 

train_ids = ids[:train]
test_ids = ids[train:train+test]
val_ids = ids[train+test:]

def write_to_csv(ids, file_name_split_csv, header =['image','mask'], is_fill=True):
  with open(file_name_split_csv, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)
    su_fill = ''
    if is_fill:
      sub_fill = '_filled'
    for id in ids:
      row0 = base_dir + 'rgb_image/' + 'SENTINEL_' + str(id) + '.jpeg'
      row1 = base_dir + 'masks' + sub_fill + '/image_binary_' + str(id) + '.png'
      rows = [row0, row1]
      csv_writer.writerow(rows)

print('writing splits to csv:', train_filename, test_filename, val_filename)

write_to_csv(train_ids, train_filename)
write_to_csv(test_ids, test_filename)
write_to_csv(val_ids, val_filename)

write_to_csv(train_ids, train_filename_fill)
write_to_csv(test_ids, test_filename_fill)
write_to_csv(val_ids, val_filename_fill)


print("Done")
