import tensor
import matplotlib.pyplot as PLT
import numpy as np

def prompt_cont():
    print('\nContinue?')
    print ('1 = Run again')
    print ('2 = New config')
    return input()

def run():
    quit = False
    new_params = True
    filename = ''
    while (quit == False):
        if new_params:
            filename = input("Config filename: ")
        parameters = []
        textfile = open('configs/'+filename)
        for line in textfile:
            parameters.append(eval(line))
        tensor.autoex(*parameters)
        ans = int(prompt_cont())
        if ans == 1:
            new_params = False
        elif ans == 2:
            new_params = True
        PLT.close('all')

if __name__ == "__main__":
    run()
