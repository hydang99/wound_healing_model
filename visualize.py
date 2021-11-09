#Add Patient
from pde.Patient import Patient 
from test_gui import App
import tkinter
from pde.param_search import param_search
from config_loader import pde_config
import threading, time
import pickle

def main():
    with open("predicted.pickle","rb") as f:
        data = pickle.load(f)
    p_list = list(data.keys())
    p_2 = []
    for p in p_list:
        p_2.append(Patient(p))
    root = tkinter.Tk()

    app = App(root, p_2)
    root.mainloop()

if __name__ == "__main__":
    main()

