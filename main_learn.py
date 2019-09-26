from src.get_arguments import train_arguments
from src.train_global import  shared_learn
if __name__ == '__main__':
    
    # this program runs the entire training process
    args = train_arguments()
    print(args)
    shared_learn(args)  
