import matplotlib.pyplot as plt
import sys
import pandas as pd

if len(sys.argv) != 3:
  print('run python plot_history.py [pandas csv log] [plot file]') 
  sys.exit()
print(sys.argv[1])
df = pd.read_csv(sys.argv[1], sep=';')
print('All keys of dataframe:', df.keys())
epoch = df['epoch']


plt.figure()
plt.title('Model dice coefficient score')
plt.xlabel('epoch')
plt.ylabel('Dice score')
train = df['score'] #train = df['dice_coef']
val = df['val_score']
#val = df['val_dice_coef']
plt.plot(epoch, train, label='train')
plt.plot(epoch, val, label='val')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(sys.argv[2] + '_score.png')
print('Plot saved at:', sys.argv[2] + '_score.png')
#plt.show()



plt.figure()
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
train = df['loss']
val = df['val_loss']
plt.plot(epoch, train, label='train')
plt.plot(epoch, val, label='val')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(sys.argv[2] + '_loss.png')
print('Plot saved at:', sys.argv[2] + '_loss.png')
