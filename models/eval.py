import matplotlib.pyplot as plt 

def plot(train_ac,val_ac):
    epochs = range(1,len(train_ac) + 1)
    fig,ax = plt.subplots(1)
    ax.plot(epochs,train_ac)
    ax.plot(epochs,val_ac)
    ax.set(xlabel='Epochs',ylabel='Accuracy')
    plt.show()
    fig.savefig('plot.png')
    return 
