from helpers.csvImporter import CSVImporter
from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from tkinter import *
from tkinter import filedialog
import tkinter as tk

class DigitGallery:

    def __init__(self, master):
        importer = CSVImporter()
        self.testData = importer.import_training_file("ressources/test.csv")

        # Global Frames
        self.master = master
        self.rootFrame = Frame(master)
        self.leftFrame = Frame(self.rootFrame)
        self.rightFrame = Frame(self.rootFrame)

        # Create Plot for the digits
        self.figure = plt.Figure(figsize=(5, 5), dpi=100)
        self.graph = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.rightFrame)

        self.result_str = StringVar()
        self.weight_matrix = []

        self.create_layout()

    def create_layout(self):

        # Load the np file
        menubar = Menu(self.master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.load_matrix)
        menubar.add_cascade(label="File", menu=filemenu)

        self.master.config(menu=menubar)

        # Left side: data selection list
        Label(self.leftFrame, text="Datenauswahl").pack(side=TOP)

        # Add scrollable list with each entry
        data_list = Listbox(self.leftFrame)

        index = 0
        for item in self.testData:
            data_list.insert(END, index)
            index += 1

        data_list.bind("<Double-Button-1>", self.list_item_selected)
        data_list.bind()
        data_list.pack(side=BOTTOM, expand=True)

        self.leftFrame.pack(side=LEFT)

        # Create result label
        result_label = Label(self.rightFrame, textvariable=self.result_str)
        result_label.config(font=("Courier", 44))
        result_label.pack(side=BOTTOM)

        self.rootFrame.pack()
        self.rootFrame.mainloop()

    def list_item_selected(self, event):
        widget = event.widget
        selection = widget.curselection()
        value = widget.get(selection[0])

        # Test the data
        network = NeuronalNetwork(
            layers=[784, 20, 10],
            weight_matrix=self.weight_matrix,
            fnc_activate_type="log"
        )
        result = network.test_single_digit(self.testData[value])

        # Turn output vector into digit
        resultDigit = -1
        max_val = 0
        for i in range(len(result)):
            if result[i] > max_val:
                max_val = result[i]
                resultDigit = i

        self.visualize(data=self.testData[value]["input"], result=resultDigit)

    def visualize(self, data, result):

        # Create plot data
        imgData = np.array(data).reshape((28, 28))

        # Remove old data
        self.graph.clear()
        self.graph.imshow(imgData, cmap=plt.cm.Greys)

        # Remove old canvas
        self.canvas.get_tk_widget().destroy()

        # Add new canvas with new data
        self.canvas = FigureCanvasTkAgg(self.figure, self.rightFrame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add button and result label
        self.result_str.set("Ergebis: " + result.__str__())
        self.rightFrame.pack()

        self.rootFrame.update_idletasks()
        self.rightFrame.update_idletasks()

    def load_matrix(self):
        file = filedialog.askopenfilename(title="Weight Matrix laden (*.np Datei)")
        self.weight_matrix = np.load(file)

root = Tk()
app = DigitGallery(root)
root.mainloop()

