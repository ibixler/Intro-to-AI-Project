import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Stand in for processing the image and returning information about style, genre, and artist
def process_img(img):
    return "Result Text Here"


def image_uploader():
    file_types = [("Png files", "*.png"), ("Jpg files", "*.jpg"), ("Jpeg files", "*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=file_types)

    # Check for file path
    if len(path):
        img = Image.open(path)
        img = img.resize((min(int(400 * img.width / img.height), 500), 400))
        pic = ImageTk.PhotoImage(img)

        label.config(image=pic)
        label.image = pic

        resultText.config(text=process_img(img))

    else:
        print("No file chosen. Please try again.")


if __name__ == "__main__":
    # App object
    app = tk.Tk()

    # App title and size
    app.title("p(AI)nt")
    app.geometry("600x600")

    # Setting background and button color
    app.option_add("*Label*Background", "white")
    app.option_add("*Button*Background", "gray")

    label = tk.Label(app)
    label.pack(pady=10)

    # Upload Button
    uploadButton = tk.Button(app, text="Locate Image", command=image_uploader)
    uploadButton.pack(side=tk.BOTTOM, pady=20)

    # Result text
    resultText = tk.Label(app)
    resultText.pack()
    # resultText.config(state=tk.DISABLED)

    app.mainloop()
