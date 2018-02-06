import tkinter as Tk
import tkinter.filedialog as filedialog
import numpy as np
from helpers.csvImporter import CSVImporter
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from tkinter import ttk
from tkinter import *
from neuronalNetwork.NeuralNetwork import NeuralNetwork


class digitRecognitionGUI:
    def __init__(self, master):
        self.master = master
        master.minsize(width=800, height=800)

        # Load the test Data from a csv File
        importer = CSVImporter()
        self.testData = importer.import_file("ressources/test.csv")

        # Config
        self.minProb = 0

        # Create the basic layout
        self.__create_init_layout()
        self.master.winfo_toplevel().title("DigitRecognizer GUI")

    def __create_init_layout(self):

        self.init_frame = Frame(self.master)

        # Configure input for the number of neurons in the hidden layer
        neuron_frame = Frame(self.init_frame)
        Label(neuron_frame, text="Anzahl der Neuronen im Hidden Layer: ").pack(side=LEFT)
        self.neuron_number_input = Entry(neuron_frame)
        self.neuron_number_input.insert(0, 20)
        self.neuron_number_input.pack(side=RIGHT)
        neuron_frame.pack()

        # Configure input for the algorithm type
        alg_frame = Frame(self.init_frame)
        Label(alg_frame, text="Verwendeter Lernalgorithmus: ").pack(side=LEFT)
        self.alg_var = StringVar(alg_frame)
        algorithms = {"log", "tanH"}
        self.alg_var.set("log")
        self.alg_menu = OptionMenu(alg_frame, self.alg_var, *algorithms)
        self.alg_menu.pack(side=RIGHT)
        alg_frame.pack()

        # Configure the file selection section
        Button(self.init_frame, text="Datei auswählen", command=self.__load_matrix).pack()
        self.init_frame.pack()

    def __create_tabs(self):
        self.init_frame.destroy()

        # Create tabbed layout
        notebook = ttk.Notebook(self.master)

        self.frame_overview = ttk.Frame(notebook)
        self.frame_visualizer = ttk.Frame(notebook)
        self.frame_tester = ttk.Frame(notebook)

        notebook.add(self.frame_overview, text="Übersicht")
        notebook.add(self.frame_visualizer, text="Visualizer")
        notebook.add(self.frame_tester, text="Tester")

        notebook.pack(expand=1, fill="both")

        # Fill the Tabs
        self.__create_overview_layout()
        self.__create_visualizer_layout()
        self.__create_tester_layout()

    def __create_overview_layout(self):
        root_frame = Frame(self.frame_overview)

        figure, str_result = self.__generate_overview_matrix()
        self.overview_result_label = Label(root_frame, text=str_result).pack(side=BOTTOM)

        self.overview_canvas = FigureCanvasTkAgg(figure, root_frame)
        self.overview_canvas.show()
        self.overview_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        configFrame = Frame(root_frame)
        self.probScale = Scale(configFrame, from_=0, to=100, orient=HORIZONTAL, command=self.__update_min_prob)
        self.probScale.set(self.minProb * 100)
        self.probScale.pack(side=LEFT)
        self.probBtn = Button(configFrame, text="Min. Wahrscheinlichkeit setzen", command=self.__update_prob).pack(side=RIGHT)
        configFrame.pack()

        root_frame.pack(fill="both", expand=True)

    def __create_visualizer_layout(self):
        root_frame = Frame(self.frame_visualizer)
        self.leftFrame_visualizer = Frame(root_frame)
        self.rightFrame_visualizer = Frame(root_frame)

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
            str = "%s - (Wert: %s)" % (index, self.__vector_to_digit(item[-10:]))
            data_list.insert(END, str)
            index += 1

        data_list.bind("<Double-Button-1>", self.__visualizer_item_selected)
        data_list.bind()
        data_list.pack(side=LEFT, expand=True, fill="both")

        self.leftFrame_visualizer.pack(side=LEFT, expand=True, fill="both")

        # Dummy data
        self.__visualize(data=np.zeros((28, 28)), result=["", ""])

        # Create result label
        result_label = Label(self.rightFrame_visualizer, textvariable=self.result_str)
        result_label.pack(side=BOTTOM)

        root_frame.pack(fill="both", expand=True)

    def __create_tester_layout(self):

        root_frame = Frame(self.frame_tester)

        # Canvas to draw on
        self.canvas = Canvas(root_frame, width=28, height=28, bg="grey")
        self.canvas.pack(expand=NO, fill=NONE, side=TOP)
        self.canvas.bind("<B1-Motion>", self.__paint)

        # Buttons to test or reset the image
        button_delete = Button(root_frame, text="Reset", command=self.__reset)
        button_delete.pack(side=BOTTOM)

        button_test = Button(root_frame, text="Testen", fg="red", command=self.__get_img)
        button_test.pack(side=BOTTOM)

        # Digit Label
        self.digit_label = Label(root_frame, text="")
        self.digit_label.pack(side=BOTTOM)

        root_frame.pack()
        root_frame.mainloop()

    def __generate_overview_matrix(self):
        data = np.zeros((10, 10))

        no_classification = 0
        correct_classification = 0
        sums = np.zeros((10, 1))
        sums_correct = np.zeros((10, 1))

        for i in range(len(self.overview_data)):

            if self.overview_data[i][0] >= 0 and self.overview_data[i][1] >= 0 and self.overview_data[i][0] < 10 and self.overview_data[i][1] < 10 and self.overview_data[i][2] >= self.minProb:

                # Rows = target, cols = actual
                data[self.overview_data[i][0]][self.overview_data[i][1]] += 1

                if self.overview_data[i][0] == self.overview_data[i][1]:
                    correct_classification += 1
                    sums_correct[self.overview_data[i][0]] += 1

                    sums[self.overview_data[i][0]] += 1
            else:
                no_classification += 1

        str_result = "Richtig klassifiziert (alle Daten): %s von %s (%s%%) \n Nicht klassifiziert: %s" \
                     % (correct_classification, len(self.overview_data),
                        (correct_classification / len(self.overview_data)) * 100,
                        no_classification)

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
        mask = np.ma.masked_where(data == 0, data)
        cmap.set_bad(color="lightgrey")
        imgplot = graph.imshow(mask, cmap=cmap)

        # Add sums to the right side
        for i in range(10):
            str = "%s / %s" % (int(sums_correct[i][0]).__str__(), int(sums[i][0]).__str__())
            graph.text(10, i, str)

        # Add text to the graph
        for i in range(10):
            for j in range(10):
                graph.text(-0.2 + i, 0.1 + j, int(data[j][i]))

        return figure, str_result

    def __visualizer_item_selected(self, event):
        widget = event.widget
        selection = widget.curselection()
        value = widget.get(selection[0])
        string_parts = value.split(" ")

        # Test the data
        result = self.network.predict(self.testData[int(float(string_parts[0]))])
        self.__visualize(data=self.testData[int(float(string_parts[0]))][:-10], result=[np.argmax(result), np.amax(result)])

    def __visualize(self, data, result):

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
        self.result_str.set("Ergebnis: %s | Wert: %s " % (result[0], result[1]))
        self.rightFrame_visualizer.pack()

        self.frame_visualizer.update_idletasks()
        self.rightFrame_visualizer.update_idletasks()

    def __get_img(self):
        width = int(self.canvas["width"])
        height = int(self.canvas["height"])
        colors = []

        for x in range(width):
            for y in range(height):
                colors.append(self.get_pixel_color(self.canvas, y, x))

        # visualizer = Visualizer()
        # visualizer.visualize(colors)

        self.__test_img(colors)

    def __test_img(self, testData):
        result = self.network.predict(testData)
        self.digit_label["text"] = ("Folgende Ziffer wurde erkannt: %s | Wert: %s" % (np.argmax(result), np.amax(result)))

    def __paint(self, event):
        python_green = "#000000"
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=python_green)

    def __reset(self):
        self.canvas.delete("all")
        self.digit_label["text"] = ""

    def __update_min_prob(self, event):
        self.minProb = int(event) / 100

    def __update_prob(self):

        for widget in self.frame_overview.winfo_children():
            widget.destroy()

        self.__create_overview_layout()

    def __load_matrix(self):
        file = filedialog.askopenfilename(title="Weight Matrix laden (*.npy Datei)")

        self.weight_matrix = np.load(file)

        self.network = NeuralNetwork(
            layers=[784, int(self.neuron_number_input.get()), 10],
            weight_matrix=self.weight_matrix,
            fnc_activate_type=self.alg_var.get()
        )
        data, _ = self.network.test(self.testData)

        self.overview_data = data
        self.__create_tabs()

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
