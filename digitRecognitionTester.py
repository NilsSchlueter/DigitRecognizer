from tkinter import *
from tkinter import filedialog
from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import numpy as np


class digitRecognitionTester:

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        # Load the np file
        menubar = Menu(master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.load_matrix)
        menubar.add_cascade(label="File", menu=filemenu)

        master.config(menu=menubar)

        # Digit Label
        self.digit_label = Label(master, text="")
        self.digit_label.pack(side=RIGHT)


        # Button to test the image
        self.button_test = Button(frame,
                             text="Testen", fg="red",
                             command=self.get_img)

        self.button_test.pack(side=RIGHT)

        self.button_delete = Button(frame,
                                    text="Reset",
                                    command=self.reset)
        self.button_delete.pack(side=RIGHT)

        # Canvas to draw on
        self.canvas = Canvas(frame,
                             width=28,
                             height=28,
                             bg="grey")
        self.canvas.pack(expand=NO, fill=NONE)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Status Label
        self.status_label = Label(master, text="NICHT Bereit - Keine Weight Matrix geladen.")
        self.status_label.pack(side=TOP)

        # Empty Weight Matrix
        self.weight_matrix = np.array([])

    def get_img(self):
        self.status_label["text"] = ""

        width = int(self.canvas["width"])
        height = int(self.canvas["height"])
        colors = []

        for x in range(width):
            for y in range(height):
                colors.append(self.get_pixel_color(self.canvas, y, x))

        #visualizer = Visualizer()
        #visualizer.visualize(colors)

        self.test_img({"input": colors})

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

    def load_matrix(self):
        file = filedialog.askopenfilename(title="Weight Matrix laden (*.np Datei)")

        self.weight_matrix = np.load(file)
        print(self.weight_matrix.shape)
        self.status_label["text"] = "Bereit!"

    def reset(self):
        self.canvas.delete("all")
        self.digit_label["text"] = ""

root = Tk()
app = digitRecognitionTester(root)
root.mainloop()

