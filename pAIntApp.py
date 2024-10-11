import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import input

def process_img(path):
    return input.process_paint(path)

def image_uploader():
    file_types = [("Png files", "*.png"), ("Jpg files", "*.jpg"), ("Jpeg files", "*.jpeg")]
    path = filedialog.askopenfilename(filetypes=file_types)
    
    if len(path):
        img = Image.open(path)
        img = img.resize((min(int(400 * img.width / img.height), 500), 400))
        pic = ImageTk.PhotoImage(img)

        label.config(image=pic)
        label.image = pic

        resultText.config(state=tk.NORMAL)  
        resultText.delete(1.0, tk.END)  
        resultText.insert(tk.END, process_img(path))  
        resultText.config(state=tk.DISABLED)  
    else:
        print("No file chosen. Please try again.")

root = tk.Tk()
root.geometry("600x900")

label = tk.Label(root)
label.pack()

text_frame = tk.Frame(root)
text_frame.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(text_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

resultText = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
resultText.pack(fill="both", expand=True)

scrollbar.config(command=resultText.yview)

upload_btn = tk.Button(root, text="Upload Image", command=image_uploader)
upload_btn.pack()

root.mainloop()
