# import tkinter as tk
# from tkinter import filedialog, Label, Button, OptionMenu
# from PIL import Image, ImageTk
# import tensorflow as tf
# import numpy as np
# from keras.models import load_model
#
# def preprocess_image(image, model_type):
#     if model_type == 'CNN':
#         IMAGE_SIZE = 200
#     elif model_type == 'ANN':
#         IMAGE_SIZE = 112
#     elif model_type == 'VGG':
#         IMAGE_SIZE = 224
#     elif model_type == 'Xception':
#         IMAGE_SIZE = 224
#     elif model_type == 'Inception':
#         IMAGE_SIZE = 224
#
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#     image /= 255.0  # normalize to [0,1] range
#     return image
#
# def load_and_preprocess_image(path, model_type):
#     image = tf.io.read_file(path)
#     return preprocess_image(image, model_type)
#
# def predict_image(model, file_path, model_type):
#     image = load_and_preprocess_image(file_path, model_type)
#     image = np.expand_dims(image, axis=0)
#     pred = model.predict(image)
#     if pred[0] > 0.5:
#         label = 'Dog nè'
#     else:
#         label = 'Cat nè'
#     return label
#
# def load_custom_model(model_path):
#     return load_model(model_path)
#
# def upload_image():
#     try:
#         file_path = filedialog.askopenfilename()
#         uploaded = Image.open(file_path)
#         uploaded.thumbnail((400, 400))
#         im = ImageTk.PhotoImage(uploaded)
#         sign_image.configure(image=im)
#         sign_image.image = im
#         label.configure(text='')
#         show_classify_button(model, file_path, model_var.get())
#     except Exception as e:
#         print(e)
#
# def classify(model, file_path, model_type):
#     label_text = predict_image(model, file_path, model_type)
#     label.configure(text=label_text, foreground='#011638')
#
# def show_classify_button(model, file_path, model_type):
#     classify_b = Button(top, text="Classify Image", command=lambda: classify(model, file_path, model_type), padx=10, pady=5)
#     classify_b.configure(background='#CDCDCD', foreground='white', font=('arial', 10, 'bold'), bd=0)
#     classify_b.place(relx=0.79, rely=0.46)
#
# def switch_model(model_name):
#     global model
#     model_path = model_paths[model_name]
#     model = load_custom_model(model_path)
#
# top = tk.Tk()
# top.geometry('800x600')
# top.title('CatsVSDogs Classification')
#
# # Set background image
# bg_image = Image.open('/home/hoangdd/Downloads/artificial_intelligence_4.jpg')
# bg_image = bg_image.resize((800, 600), Image.LANCZOS)
# bg_photo = ImageTk.PhotoImage(bg_image)
#
# bg_label = Label(top, image=bg_photo)
# bg_label.place(relwidth=1, relheight=1)
#
# # Create a frame for the header
# header_frame = tk.Frame(top, bg='#CDCDCD', bd=0, highlightthickness=0)
# header_frame.pack(fill=tk.X, pady=10)
#
# # Add heading label
# heading = Label(header_frame, text="CatsVSDogs Classification", pady=20, font=('arial', 24, 'bold'))
# heading.configure(background='#CDCDCD', foreground='#364156')
# heading.pack()
#
# # Create a frame for the model selection
# model_frame = tk.Frame(top, bg='#CDCDCD', bd=0, highlightthickness=0)
# model_frame.pack(fill=tk.X, pady=10)
#
# # Dropdown menu for selecting model
# model_var = tk.StringVar(top)
# model_var.set('CNN')  # Default model is CNN
# model_paths = {
#     'CNN': r'/home/hoangdd/Downloads/jupyter/best_cnn_model_50epoch.h5',
#     'ANN': r'/home/hoangdd/Downloads/jupyter/ann_model.ipynb.h5',
#     'VGG': r'/home/hoangdd/Downloads/jupyter/vgg_model.h5',
#     'Inception': r'/home/hoangdd/Downloads/jupyter/inception_model.h5',
#     'Xception': r'/home/hoangdd/Downloads/jupyter/xception_model.h5'
# }
# model_menu = OptionMenu(model_frame, model_var, *model_paths.keys(), command=switch_model)
# model_menu.configure(background='#CDCDCD', foreground='black', font=('arial', 12, 'bold'), bd=0, highlightthickness=0)
# model_menu.pack(side=tk.LEFT, padx=20)
#
# # Create a frame for the main content
# content_frame = tk.Frame(top, bg='#CDCDCD', bd=0, highlightthickness=0)
# content_frame.pack(expand=True, fill=tk.BOTH)
#
# # Display area for the uploaded image
# sign_image = Label(content_frame, bg='#CDCDCD')
# sign_image.pack(expand=True, pady=20)
#
# # Label for classification result
# label = Label(content_frame, background='#CDCDCD', font=('arial', 15, 'bold'))
# label.pack()
#
# # Create a frame for the upload button
# button_frame = tk.Frame(top, bg='#CDCDCD', bd=0, highlightthickness=0)
# button_frame.pack(fill=tk.X, pady=20)
#
# # Upload button
# upload = Button(button_frame, text="Upload an image", command=upload_image, padx=20, pady=10)
# upload.configure(background='#CDCDCD', foreground='black', font=('arial', 12, 'bold'), bd=0, highlightthickness=0)
# upload.pack()
#
# # Load the default model (CNN)
# model = load_custom_model(model_paths[model_var.get()])
#
# # Start the main loop
# top.mainloop()


# import tkinter as tk
# from tkinter import filedialog, Label, Button, OptionMenu, StringVar
# from PIL import Image, ImageTk
# import tensorflow as tf
# import numpy as np
# from keras.models import load_model
#
# # Function to preprocess image
# def preprocess_image(image, model_type):
#     IMAGE_SIZE = {
#         'CNN': 200,
#         'ANN': 112,
#         'VGG': 224,
#         'Xception': 224,
#         'Inception': 224
#     }.get(model_type, 224)
#
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#     image /= 255.0  # normalize to [0,1] range
#     return image
#
# # Function to load and preprocess image
# def load_and_preprocess_image(path, model_type):
#     image = tf.io.read_file(path)
#     return preprocess_image(image, model_type)
#
# # Function to predict image
# def predict_image(model, file_path, model_type):
#     image = load_and_preprocess_image(file_path, model_type)
#     image = np.expand_dims(image, axis=0)
#     pred = model.predict(image)
#     confidence = pred[0][0] if pred.shape[-1] == 1 else np.max(pred[0])
#     label = 'Dog' if confidence > 0.5 else 'Cat'
#     return label, confidence
#
# # Function to load custom model
# def load_custom_model(model_path):
#     return load_model(model_path)
#
# # Function to upload image
# def upload_image():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         uploaded = Image.open(file_path)
#         uploaded.thumbnail((400, 400))
#         im = ImageTk.PhotoImage(uploaded)
#         sign_image.config(image=im)
#         sign_image.image = im
#         label.config(text='')
#         classify_button.config(state=tk.NORMAL, command=lambda: classify_image(file_path))
#
# # Function to classify image
# def classify_image(file_path):
#     label_text, confidence = predict_image(model, file_path, model_var.get())
#     label.config(text=f'{label_text} ({confidence:.2%} confidence)', foreground='#011638')
#
# # Function to switch model
# def switch_model(model_name):
#     global model
#     model_path = model_paths[model_name]
#     model = load_custom_model(model_path)
#
# # Initialize main window
# top = tk.Tk()
# top.geometry('800x650')
# top.title('CatsVSDogs Classification')
# top.configure(background='#F0F0F0')
#
# # Create a frame for the header
# header_frame = tk.Frame(top, bg='#364156', pady=10)
# header_frame.pack(fill=tk.X)
#
# # Add heading label
# heading = Label(header_frame, text="CatsVSDogs Classification", pady=20, font=('arial', 24, 'bold'), bg='#364156', fg='white')
# heading.pack()
#
# # Create a frame for the description
# description_frame = tk.Frame(top, bg='#F0F0F0', pady=10)
# description_frame.pack(fill=tk.X, padx=20)
#
# description = Label(description_frame, text="This application allows you to classify images of cats and dogs using different models. "
#                                             "Upload an image, select a model, and get the prediction with confidence score.",
#                     font=('arial', 14), bg='#F0F0F0', wraplength=760, justify=tk.LEFT)
# description.pack()
#
# # Create a frame for the model selection
# model_frame = tk.Frame(top, bg='#F0F0F0', pady=10)
# model_frame.pack(fill=tk.X, padx=20)
#
# model_label = Label(model_frame, text="Select Model:", font=('arial', 12, 'bold'), bg='#F0F0F0')
# model_label.pack(side=tk.LEFT)
#
# model_var = StringVar(top)
# model_var.set('CNN')  # Default model is CNN
# model_paths = {
#     'CNN': r'/home/hoangdd/Downloads/jupyter/best_cnn_model_50epoch.h5',
#     'ANN': r'/home/hoangdd/Downloads/jupyter/ann_model.ipynb.h5',
#     'VGG': r'/home/hoangdd/Downloads/jupyter/vgg_model.h5',
#     'Inception': r'/home/hoangdd/Downloads/jupyter/inception_model.h5',
#     'Xception': r'/home/hoangdd/Downloads/jupyter/xception_model.h5'
# }
# model_menu = OptionMenu(model_frame, model_var, *model_paths.keys(), command=switch_model)
# model_menu.config(bg='#364156', fg='white', font=('arial', 12, 'bold'))
# model_menu.pack(side=tk.LEFT, padx=20)
#
# # Create a frame for the main content
# content_frame = tk.Frame(top, bg='#E0E0E0', pady=10, padx=10)
# content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
#
# # Display area for the uploaded image
# sign_image = Label(content_frame, bg='#E0E0E0')
# sign_image.pack(expand=True, pady=20)
#
# # Label for classification result
# label = Label(content_frame, text="", bg='#E0E0E0', font=('arial', 18, 'bold'))
# label.pack()
#
# # Create a frame for the buttons
# button_frame = tk.Frame(top, bg='#E0E0E0', pady=10, padx=10)
# button_frame.pack(fill=tk.X, padx=20, pady=10)
#
# # Upload button
# upload_button = Button(button_frame, text="Upload an Image", command=upload_image, padx=20, pady=10, bg='#364156', fg='white', font=('arial', 12, 'bold'))
# upload_button.pack(side=tk.LEFT, padx=10)
#
# # Classify button (initially disabled)
# classify_button = Button(button_frame, text="Classify Image", padx=20, pady=10, bg='#364156', fg='white', font=('arial', 12, 'bold'), state=tk.DISABLED)
# classify_button.pack(side=tk.LEFT, padx=10)
#
# # Load the default model (CNN)
# model = load_custom_model(model_paths[model_var.get()])
#
# # Start the main loop
# top.mainloop()

import tkinter as tk
from tkinter import filedialog, Label, Button, OptionMenu, StringVar, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from keras.models import load_model

# Function to preprocess image
def preprocess_image(image, model_type):
    IMAGE_SIZE = {
        'CNN': 200,
        'ANN': 112,
        'VGG': 224,
        'Xception': 224,
        'Inception': 224
    }.get(model_type, 224)

    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image

# Function to load and preprocess image
def load_and_preprocess_image(path, model_type):
    image = tf.io.read_file(path)
    return preprocess_image(image, model_type)

# Function to predict image
def predict_image(model, file_path, model_type):
    image = load_and_preprocess_image(file_path, model_type)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    confidence = pred[0][0] if pred.shape[-1] == 1 else np.max(pred[0])
    label = 'Dog' if confidence > 0.5 else 'Cat'
    return label, confidence

# Function to load custom model
def load_custom_model(model_path):
    return load_model(model_path)

# Function to upload image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        uploaded = Image.open(file_path)
        uploaded.thumbnail((400, 400))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.config(image=im)
        sign_image.image = im
        label.config(text='')
        classify_button.config(state=tk.NORMAL, command=lambda: classify_image(file_path))

# Function to classify image
def classify_image(file_path):
    label_text, confidence = predict_image(model, file_path, model_var.get())
    label.config(text=f'{label_text} ({confidence:.2%} confidence)', foreground='#011638')

# Function to switch model
def switch_model(model_name):
    global model
    model_path = model_paths[model_name]
    model = load_custom_model(model_path)
    heading.config(text=f"CatsVSDogs Classification - {model_name}")

# Function to show instruction
def show_instruction():
    messagebox.showinfo("Instruction", "This application allows you to classify images of cats and dogs using different models. "
                                       "Upload an image, select a model, and get the prediction with confidence score.")

# Initialize main window
top = tk.Tk()
top.geometry('1000x700')
top.title('CatsVSDogs Classification')
top.configure(background='#F0F0F0')

# Create a frame for the header
header_frame = tk.Frame(top, bg='#364156', pady=10)
header_frame.pack(fill=tk.X)

# Add heading label
heading = Label(header_frame, text="CatsVSDogs Classification - CNN", pady=20, font=('arial', 24, 'bold'), bg='#364156', fg='white')
heading.pack()

# Add instruction button to header
instruction_button = Button(header_frame, text="Instruction", command=show_instruction, padx=10, pady=5, bg='#E0E0E0', fg='#364156', font=('arial', 10, 'bold'))
instruction_button.pack(side=tk.LEFT, padx=10)

# Create a sidebar frame
sidebar_frame = tk.Frame(top, bg='white', width=250, pady=20)
sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)

# Add model selection to sidebar
model_label = Label(sidebar_frame, text="Select Model:", font=('arial', 12, 'bold'), bg='white', fg='#364156')
model_label.pack(pady=(10, 5))

model_var = StringVar(top)
model_var.set('CNN')  # Default model is CNN
model_paths = {
    'CNN': r'/home/hoangdd/Downloads/jupyter/best_cnn_model_50epoch.h5',
    'ANN': r'/home/hoangdd/Downloads/jupyter/ann_model.ipynb.h5',
    'VGG': r'/home/hoangdd/Downloads/jupyter/vgg_model.h5',
    'Inception': r'/home/hoangdd/Downloads/jupyter/inception_model.h5',
    'Xception': r'/home/hoangdd/Downloads/jupyter/xception_model.h5'
}

model_menu = OptionMenu(sidebar_frame, model_var, *model_paths.keys(), command=switch_model)
model_menu.config(bg='#E0E0E0', fg='#364156', font=('arial', 12, 'bold'))
model_menu.pack(pady=(0, 10))

# Create a frame for the main content
content_frame = tk.Frame(top, bg='#E0E0E0', pady=10, padx=10)
content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20, side=tk.RIGHT)

# Display area for the uploaded image
sign_image = Label(content_frame, bg='#E0E0E0')
sign_image.pack(expand=True, pady=20)

# Label for classification result
label = Label(content_frame, text="", bg='#E0E0E0', font=('arial', 18, 'bold'))
label.pack()

# Add upload button to content frame (centered)
upload_button = Button(content_frame, text="Upload an Image", command=upload_image, padx=20, pady=10, bg='#E0E0E0', fg='#364156', font=('arial', 12, 'bold'))
upload_button.pack(side=tk.LEFT, pady=10, padx=(20, 10))

# Add classify button to content frame (centered)
classify_button = Button(content_frame, text="Classify Image", padx=20, pady=10, bg='#E0E0E0', fg='#364156', font=('arial', 12, 'bold'), state=tk.DISABLED)
classify_button.pack(side=tk.LEFT, pady=10, padx=(10, 20))

# Load the default model (CNN)
model = load_custom_model(model_paths[model_var.get()])

# Start the main loop
top.mainloop()
