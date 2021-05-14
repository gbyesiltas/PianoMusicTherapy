import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import madmom
import wave


import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfile
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from md_calc import calculate_md
from md_calc import export_as_midi

global time1, time2, audio_file


def popupmsg(msg):
    popup = tk.Toplevel()
    popup.geometry("150x60")
    popup.wm_title("File opened")
    label = tk.Label(popup, text=msg)
    label.pack(side="top", fill="x")
    b1 = tk.Button(popup, text="Ok", command=popup.destroy)
    b1.pack()
    popup.mainloop()

class AppMIR(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, bd=3)
        self.master.title("MIR Music Therapy")
        self.style = ttk.Style(self)
        self.style.theme_use('default')
        self.pack(fill=tk.BOTH, expand=1)
        self.parent = parent
        self.grid()
        self.timestamp_1 = tk.StringVar()
        self.timestamp_2 = tk.StringVar()
        # self.createButtons()
        self.createMenu()
        self.createInformation()
        self.createTextEntry()
        self.createComment()

    def createMenu(self):
        self.parent.title("MIR Music Therapy")
        self.pack(fill=tk.BOTH, expand=1)

        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = tk.Menu(menubar)
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="Open file...", command=self.openFile)
        fileMenu.add_command(label="Export comment...", command=self.saveComment)
        fileMenu.add_command(label="Quit", command=self.quit)

        processMenu = tk.Menu(menubar)
        menubar.add_cascade(label="Process...", menu=processMenu)
        processMenu.add_command(label="Show waveform", command=self.processFile)

        processMenu.add_command(label="Calculate metrical deviation", command=self.md_calc)

    def md_calc(self):
        # code for metrical deviation
        if(self.audio_file is not None):
            calculate_md(self.audio_file,[self.time1,self.time2])
        else:
            popupmsg('You have not loaded an audio file yet')
        
    def export_midi(self):
        # code for exporting midi using madmom cnn piano transcription
        if(self.audio_file is not None):
            export_as_midi(audio_file)
        else:
            popupmsg('You have not loaded an audio file yet')

    def createInformation(self):
        explanation = """Use the File menu to open a .wav file. 
        Use the Process menu to show the waveform of the file or 
        to get the metrical deviation over time. All graphs will pop-up
        in a new save an can be saved as an image file. Comments can be 
        exported as .txt files."""


        infolabel = tk.Label(root,
                     compound=tk.CENTER,
                     text=explanation)
        infolabel.place(x=0, y=0)

    def createComment(self):
        commentframe = tk.Frame()
        l = tk.Label(text="Add a comment")
        l.pack(side=tk.LEFT, padx=5, pady=5)
        self.commentbox = tk.Text(height=5, width=200)
        self.commentbox.pack()
        # scroll = tk.Scrollbar(self.commentbox)
        # self.commentbox.configure(yscrollcommand=scroll.set)
        # scroll.config(command=self.commentbox.yview)
        # scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def submit(self):
        self.time1 = self.timestamp_1.get()
        self.time2 = self.timestamp_2.get()

        print(self.time1, self.time2)
        self.timestamp_1.set("")
        self.timestamp_2.set("")

    def createTextEntry(self):
        # creating a label for
        # name using widget Label
        time1_label = tk.Label(root, text='Timestamp beginning part B (mm:ss)')

        # creating a entry for input
        # name using widget Entry
        time1_entry = tk.Entry(root, textvariable=self.timestamp_1)

        # creating a label for password
        time2_label = tk.Label(root, text='Timestamp end part B (mm:ss)')

        # creating a entry for password
        time2_entry = tk.Entry(root, textvariable=self.timestamp_2,)

        # creating a button using the widget
        # Button that will call the submit function
        sub_btn = tk.Button(root, text='Submit', command=self.submit)

        time1_label.pack()
        time1_entry.pack()
        time2_label.pack()
        time2_entry.pack()
        sub_btn.pack()


    def openFile(self):
        # global filename
        app.filename = askopenfilename(initialdir="/", title="Select file", filetypes=(("WAV files", "*.wav"),
                                                                                             ("All files", "*.*")))
        if app.filename is not None:
            self.audio_file = app.filename
            print(app.filename)
            popupmsg("File opened")
            
        if app.filename is None:
            popupmsg("No file selected")
        

    def saveComment(self):
        savefile = asksaveasfile(mode='w', defaultextension=".txt")
        comment = str(self.commentbox.get(1.0, tk.END))
        savefile.write(comment)
        savefile.close()

    def processFile(self):
        sig = wave.open(app.filename)
        signal = sig.readframes(-1)
        signal = np.frombuffer(signal, dtype=int)
        fs = sig.getframerate()
        time = np.linspace(0, len(signal)/ fs, num=len(signal))

        root2 = tk.Toplevel()
        fig2 = plt.Figure()
        canvas2 = FigureCanvasTkAgg(fig2, master=root2)
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
        ax = fig2.add_subplot(111)
        ax.plot(time, signal)
        canvas2.draw()
        # canvas2.show()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas2,
                                       root2)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        canvas2.get_tk_widget().place()

root = tk.Tk()

app = AppMIR(root)
root.geometry("600x400")
root.mainloop()


