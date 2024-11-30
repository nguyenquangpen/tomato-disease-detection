"""
@author: Quang Nguyen <nguyenquangpen@gmail.com>
"""

from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from Test_Model import predictTomato

def image_display_Predict(label, result_type):
    global file_path
    if result_type == "image":
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            initialdir="/mnt/d/DataDeepLearning/TestTomato",
            filetypes=[("All Files", "*.*")]
        )
        if file_path:
            try:
                image = Image.open(file_path)
                resized_image = image.resize((350, 350))
                img_tk = ImageTk.PhotoImage(resized_image)

                label.config(image=img_tk)
                label.image = img_tk
            except Exception as e:
                print(f"Lỗi khi mở ảnh: {e}")

    if result_type == 'result':
        results = predictTomato(file_path)
        label.config(text=str(results))

        with open("SolutionTomato.txt", "r") as f:
            text = f.read()
            sections = text.split('\n\n')
            for section in sections:
                if section.startswith(results):
                    text_widget.config(state="normal")
                    text_widget.delete(1.0, tk.END)
                    text_widget.insert(tk.END, section[len(results):].strip())
                    text_widget.config(state="disabled")
                    break

if __name__ == '__main__':
    win = Tk()
    win.title("Image Classification")
    win.geometry('850x510')
    win['bg'] = 'white'
    win.attributes('-topmost', True)

    # Title centered
    title = Label(win, text="Tomato Disease Detection", font=("Arial Bold", 20, 'bold'), bg='white')
    title.place(relx=0.5, y=30, anchor='n')  # Centering the title

    # Label for image (left)
    image_label_left = Label(win, bg="gray")
    image_label_left.place(x=20, y=100, width=350, height=350)

    # text for predict (right)
    text_widget = tk.Text(win, wrap="word", height=20, width=50, state="disabled")
    text_widget.pack(side="left", fill="both", expand=True)
    text_widget.place(x=400, y=200, width=400, height=250)

    # scrollbar
    scrollbar = tk.Scrollbar(win, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    text_widget.config(yscrollcommand=scrollbar.set)

    # Import button
    import_button = Button(win, text="Import Image", command=lambda: image_display_Predict(image_label_left, "image"))
    import_button.place(x=50, y=460)

    # Predict button
    predict_button = Button(win, text="Predict", command=lambda: image_display_Predict(result, "result"))
    predict_button.place(x=200, y=460)

    # title name of the disease
    disease_name = Label(win, text="Probability ", font=("Arial Bold", 15, 'bold'), bg='white')
    disease_name.place(x=550, y=100)

    # result of the disease
    result = Label(win, font=("Arial Bold", 13, 'bold'))
    result.place(x=400, y=150, width=400, height=40)

    win.mainloop()