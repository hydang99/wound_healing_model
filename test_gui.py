import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot
import matplotlib.pyplot as plt
from utils.utils import get_logger
import threading
logger = get_logger("VISUAL")
class App(threading.Thread):
    def __init__(self, master, patients, **kwargs):
        # Create a container
        frame = tkinter.Frame(master)
        # Create 2 buttons
        self.button_left = tkinter.Button(frame,text="< Last Patient",
                                        command=self.decrease,
                                            relief="groove",
                                            compound=tkinter.CENTER,
                                            bg="white",
                                            fg="black",
                                            activeforeground="blue",
                                            activebackground="white",
                                            font="arial 30",)
        self.button_left.pack(side="left")
        self.button_right = tkinter.Button(frame,text="Next Patient >",
                                        command=self.increase,
                                            relief="groove",
                                            compound=tkinter.CENTER,
                                            bg="white",
                                            fg="black",
                                            activeforeground="blue",
                                            activebackground="white",
                                            font="arial 30",)
        self.button_right.pack(side="left")
        self.button_quit = tkinter.Button(frame, text="Quit Frame",
                                        command=frame.quit)
        self.button_quit.pack(side=tkinter.BOTTOM)
        self.index = 0
        self.fig = Figure(figsize=(10, 10))
        self.patients = patients
        #self.ax = self.fig.add_subplot(111)
        # self.line, = ax.plot(range(10))

        #self.show(self.patients[0])

        self.canvas = FigureCanvasTkAgg(self.fig,master=master)
        self.increase()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()
        # threading.Thread.__init__(self)


    def decrease(self):
        logger.info("PREVIOUS PATIENT")
        self.fig.clf()
        self.button_right["state"] = "active"

        index = self.index
        self.show(self.patients[index])
        if index == 0: 
            self.button_left["state"] = "disabled"
            self.index = 1
        else: 
            self.index-=1
        self.canvas.draw()

        

    def increase(self):
        logger.info("NEXT PATIENT")
        self.fig.clf()
        index = self.index
        if index == 0: 
            self.button_left["state"] = "disabled"
        else:
            self.button_left["state"] = "active"
        self.show(self.patients[index])

        if index == len(self.patients) - 1: 
            self.button_right["state"] = "disabled"
            self.index = len(self.patients) - 2
        else:
            self.index +=1
        self.canvas.draw()

    """
    Methods for showing list of grays images
    """
    def show(self,patient):
        patient_info = patient.patient_info
        list_img, patient_id = patient_info["wound_img"], patient_info["id"]
        w=10
        h=10
        #fig=plt.figure(figsize=(20, 20))
        columns = 5
        rows = 5
        i = 1
        while i <= (min(columns*rows+1,len(list_img))):
            img = list_img[list((list_img.keys()))[i-1]]
            self.fig.add_subplot(rows,columns,i,title="{}".format(list((list_img.keys()))[i-1])).imshow(img)
            i+=1
        self.fig.suptitle("Patient {}".format(patient_id),fontsize = 16)
        self.fig.tight_layout()

