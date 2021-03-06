import tkinter as tk

from src.main_application import MainApplication


def main():
    title = "Clinical Notes Regular Expression Tool"
    root = tk.Tk()
    root.geometry('{}x{}'.format(1200, 700))
    root.title(title)
    MainApplication(root).grid(column=0, row=0)
    root.mainloop()


if __name__ == '__main__':
    main()
