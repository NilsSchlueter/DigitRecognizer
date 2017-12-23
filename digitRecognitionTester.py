from tkinter import *
from tkinter import filedialog
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
        self.digit_label = Label(master, text="Folgende Ziffer wurde erkannt: -")
        self.digit_label.pack(side=RIGHT)


        # Button to test the image
        self.button_test = Button(frame,
                             text="Testen", fg="red",
                             command=self.get_img)

        self.button_test.pack(side=RIGHT)

        # Canvas to draw on
        self.canvas = Canvas(frame,
                             width=280,
                             height=280,
                             bg="grey")
        self.canvas.pack(expand=NO, fill=NONE)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Status Label
        self.status_label = Label(master, text="NICHT Bereit - Keine Weight Matrix geladen.")
        self.status_label.pack(side=TOP)

        # Empty Weight Matrix
        self.weight_matrix = np.array([])

    def get_img(self):
        print("Getting image")

    def paint(self, event):
        python_green = "#000000"
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=python_green)

    def load_matrix(self):
        file = filedialog.askopenfilename(title="Weight Matrix laden (*.np Datei)")

        self.weight_matrix = np.load(file)
        self.status_label["text"] = "Bereit!"


root = Tk()
app = digitRecognitionTester(root)
root.mainloop()

