import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import input

def process_img(path):
    # Dummy function for illustration
    return input.process_paint(path)

def image_uploader():
    file_types = [("Png files", "*.png"), ("Jpg files", "*.jpg"), ("Jpeg files", "*.jpeg")]
    path = filedialog.askopenfilename(filetypes=file_types)

    # Check for file path
    if len(path):
        img = Image.open(path)
        img = img.resize((min(int(400 * img.width / img.height), 500), 400))
        pic = ImageTk.PhotoImage(img)

        label.config(image=pic)
        label.image = pic

        # Clear previous text and insert new text
        resultText.config(state=tk.NORMAL)  # Enable editing to insert text
        resultText.delete(1.0, tk.END)  # Clear previous content
        resultText.insert(tk.END, process_img(path))  # Insert new content
        resultText.config(state=tk.DISABLED)  # Disable editing again to make it read-only

    else:
        print("No file chosen. Please try again.")

# GUI setup
root = tk.Tk()
root.geometry("600x900")

# Image label to show selected image
label = tk.Label(root)
label.pack()

# Frame to contain the Text widget and Scrollbar
text_frame = tk.Frame(root)
text_frame.pack(fill="both", expand=True)

# Scrollable Text widget for result display
scrollbar = tk.Scrollbar(text_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

resultText = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
resultText.pack(fill="both", expand=True)

# Configure the scrollbar to work with the Text widget
scrollbar.config(command=resultText.yview)

# Add button to upload image
upload_btn = tk.Button(root, text="Upload Image", command=image_uploader)
upload_btn.pack()

root.mainloop()
