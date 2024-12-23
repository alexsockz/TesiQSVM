from tkinter import filedialog
import os
directory=filedialog.askdirectory()

NFILE=len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])-1

inp = input("inserisci numero di partenza")

for i in reversed(range(NFILE)):
    nxt=i+int(inp)
    os.rename(directory+'/data'+str(i)+".csv",directory+'/data'+str(nxt)+".csv")