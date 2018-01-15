import tkinter as Tk
import tkinter.filedialog as filedialog
import numpy as np
from helpers.csvImporter import CSVImporter
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib import colors
from tkinter import ttk
from tkinter import *
from neuronalNetwork.neuronalNetwork import NeuronalNetwork


class digitRecognitionGUI:

    def __init__(self, master):
        self.master = master
        master.minsize(width=600, height=600)
        master.maxsize(width=600, height=600)

        # Load the test Data from a csv File
        importer = CSVImporter()
        self.testData = importer.import_training_file("ressources/test.csv")

        # Create the basic layout
        self.__create_layout()

    def __create_layout(self):

        # Create file menu to load the weight matrix
        menubar = Menu(self.master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.load_matrix)
        menubar.add_cascade(label="File", menu=filemenu)

        self.master.config(menu=menubar)

        # Create tabbed layout
        notebook = ttk.Notebook(self.master)

        self.frame_overview = ttk.Frame(notebook)
        self.frame_visualizer = ttk.Frame(notebook)
        self.frame_tester = ttk.Frame(notebook)

        notebook.add(self.frame_overview, text="Übersicht")
        notebook.add(self.frame_visualizer, text="Visualizer")
        notebook.add(self.frame_tester, text="Tester")

        notebook.pack(expand=1, fill="both")

        # Fill each Tab
        self.__create_visualizer_layout()
        self.__create_tester_layout()

    def __create_overview_layout(self, overview_data):
        root_frame = Frame(self.frame_overview)

        data = np.zeros((10, 10))

        no_classification = 0
        correct_classification = 0
        for i in range(len(overview_data)):

            if overview_data[i][0] >= 0 and overview_data[i][1] >= 0 and overview_data[i][0] < 10 and overview_data[i][1] < 10:
                # Rows = target, cols = actual
                data[overview_data[i][0]][overview_data[i][1]] += 1

                if overview_data[i][0] == overview_data[i][1]:
                    correct_classification += 1
            else:
                no_classification += 1


        str_result = "Richtig klassifiziert: %s von %s (%s%%) | Nicht klassifiziert: %s" \
                     % (correct_classification, len(overview_data), (correct_classification / len(overview_data)) * 100, no_classification)
        Label(root_frame, text=str_result).pack(side=BOTTOM)

        # Creating the figure
        figure = plt.Figure(figsize=(7, 7), dpi=100)
        graph = figure.add_subplot(111)

        major_ticks = np.arange(0, 10, 1)
        graph.set_xticks(major_ticks)
        graph.set_yticks(major_ticks)
        graph.set_xticklabels(major_ticks)
        graph.set_yticklabels(major_ticks)

        graph.set_xlabel("Predicted")
        graph.set_ylabel("Target")

        # Configure color
        cmap = plt.cm.RdYlGn
        cmap.set_under(color="lightgray")
        imgplot = graph.imshow(data, cmap=cmap, vmin=1)

        # Add text to the graph
        for i in range(10):
            for j in range(10):
                graph.text(-0.2 + i, 0.1 + j, data[j][i])


        canvas = FigureCanvasTkAgg(figure, root_frame)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        root_frame.pack()

    def __create_visualizer_layout(self):
        rootFrame = Frame(self.frame_visualizer)
        self.leftFrame_visualizer = Frame(rootFrame)
        self.rightFrame_visualizer = Frame(rootFrame)

        self.figure_visualizer = plt.Figure(figsize=(5, 5), dpi=100)
        self.graph_visualizer = self.figure_visualizer.add_subplot(111)
        self.canvas_visualizer = FigureCanvasTkAgg(self.figure_visualizer, self.rightFrame_visualizer)

        self.result_str = StringVar()

        # Left side: data selection list
        Label(self.leftFrame_visualizer, text="Datenauswahl").pack(side=TOP)

        # Add scrollable list with each entry
        data_list = Listbox(self.leftFrame_visualizer)

        index = 0
        for item in self.testData:
            str = "%s - (Wert: %s)" % (index, self.__vector_to_digit(item["output"]))
            data_list.insert(END, str)
            index += 1

        data_list.bind("<Double-Button-1>", self.__visualizer_item_selected)
        data_list.bind()
        data_list.pack(side=LEFT, expand=True, fill="both")

        self.leftFrame_visualizer.pack(side=LEFT, expand=True, fill="both")

        # Dummy data
        self.visualize(data=np.zeros((28,28)), result="")

        # Create result label
        result_label = Label(self.rightFrame_visualizer, textvariable=self.result_str)
        result_label.config(font=("Courier", 44))
        result_label.pack(side=BOTTOM)

        rootFrame.pack()

    def __create_tester_layout(self):

        root_frame = Frame(self.frame_tester)

        # Canvas to draw on
        self.canvas = Canvas(root_frame,
                             width=28,
                             height=28,
                             bg="grey")
        self.canvas.pack(expand=NO, fill=NONE, side=TOP)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons to test or reset the image
        button_delete = Button(root_frame, text="Reset", command=self.reset)
        button_delete.pack(side=BOTTOM)

        button_test = Button(root_frame, text="Testen", fg="red",command=self.get_img)
        button_test.pack(side=BOTTOM)

        # Digit Label
        self.digit_label = Label(root_frame, text="")
        self.digit_label.pack(side=BOTTOM)

        root_frame.pack()
        root_frame.mainloop()

    def __visualizer_item_selected(self, event):
        widget = event.widget
        selection = widget.curselection()
        value = widget.get(selection[0])
        string_parts = value.split(" ")

        # Test the data
        result = self.network.test_single_digit(self.testData[int(float(string_parts[0]))])

        # Turn output vector into digit
        resultDigit = -1
        max_val = 0
        for i in range(len(result)):
            if result[i] > max_val:
                max_val = result[i]
                resultDigit = i

        self.visualize(data=self.testData[int(float(string_parts[0]))]["input"], result=resultDigit)

    def visualize(self, data, result):

        # Create plot data
        imgData = np.array(data).reshape((28, 28))

        # Remove old data
        self.graph_visualizer.clear()
        self.graph_visualizer.imshow(imgData, cmap=plt.cm.Greys)

        # Remove old canvas
        self.canvas_visualizer.get_tk_widget().destroy()

        # Add new canvas with new data
        self.canvas_visualizer = FigureCanvasTkAgg(self.figure_visualizer, self.rightFrame_visualizer)
        self.canvas_visualizer.show()
        self.canvas_visualizer.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        # Add button and result label
        self.result_str.set("Ergebis: " + result.__str__())
        self.rightFrame_visualizer.pack()

        self.frame_visualizer.update_idletasks()
        self.rightFrame_visualizer.update_idletasks()

    def get_img(self):
        width = int(self.canvas["width"])
        height = int(self.canvas["height"])
        colors = []

        for x in range(width):
            for y in range(height):
                colors.append(self.get_pixel_color(self.canvas, y, x))

        # visualizer = Visualizer()
        # visualizer.visualize(colors)

        self.test_img({"input": colors})

    def test_img(self, testData):
        network = NeuronalNetwork(
            layers=[784, 20, 10],
            weight_matrix=self.weight_matrix,
            fnc_activate_type="log"
        )
        result = network.test_single_digit(testData)
        print(result)

        resultDigit = -1
        max_val = 0
        for i in range(len(result)):
            if result[i] > max_val:
                max_val = result[i]
                resultDigit = i

        self.digit_label["text"] = ("Folgende Ziffer wurde erkannt: %s" % resultDigit)

    def paint(self, event):
        python_green = "#000000"
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=python_green)

    def reset(self):
        self.canvas.delete("all")
        self.digit_label["text"] = ""

    def load_matrix(self):
        file = filedialog.askopenfilename(title="Weight Matrix laden (*.np Datei)")
        self.weight_matrix = np.load(file)
        self.network = NeuronalNetwork(
            layers=[784, 20, 10],
            weight_matrix=self.weight_matrix,
            fnc_activate_type="log"
        )
        str, data = self.network.test(self.testData)

        self.__create_overview_layout(data)

    @staticmethod
    def __vector_to_digit(vector):
        digit = 0
        for i in range(len(vector)):
            if vector[i] != 0:
                digit += i
        return digit

    @staticmethod
    def get_pixel_color(canvas, x, y):
        ids = canvas.find_overlapping(x, y, x, y)

        if len(ids) > 0:
            index = ids[-1]
            color = canvas.itemcget(index, "fill")
            color = color.upper()
            if color != '':
                return 1

        return 0

root = Tk()

app = digitRecognitionGUI(root)
root.mainloop()
