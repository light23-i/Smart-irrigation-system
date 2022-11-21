import matplotlib.pyplot as plt 

def plot(train_loss,val_loss):
    epochs = range(1,len(train_loss) + 1)
    fig,ax = plt.subplots(1)
    ax.plot(epochs,train_loss)
    ax.plot(epochs,val_loss)
    ax.set(xlabel='Epochs',ylabel='Loss')
    plt.show()
    fig.savefig('plot.png')
    return 
