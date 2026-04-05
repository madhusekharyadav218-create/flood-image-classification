from flask import Flask, url_for, redirect, render_template, request, session
import mysql.connector, os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'admin'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3307",
    database='aerial'
)
# import pymysql

# mydb = pymysql.connect(
#     host="localhost",
#     user="root",
#     password="root",
#     port=3306,
#     database='aerial'
# )

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('home.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


# Define the DenseNet model
class DenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.feature_maps = output

    def generate_cam(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        one_hot = torch.zeros((1, output.size(-1)), dtype=torch.float32).to(input_tensor.device)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = torch.autograd.grad(outputs=output[:, class_idx],
                                        inputs=self.feature_maps,
                                        grad_outputs=torch.ones_like(output[:, class_idx]),
                                        retain_graph=True)[0]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (input_tensor.size(-1), input_tensor.size(-2)))
        return cam

# Initialize model and load weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DenseNetModel(num_classes=6).to(device)
model.load_state_dict(torch.load("densenet.pt", map_location=device))
model.eval()

# Define class names
class_names = ['Building', 'flooded', 'forest', 'mountains', 'sea', 'street']

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = image_transform(image).unsqueeze(0).to(device)
    return input_tensor, image

@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Check if a file is uploaded
        if 'file' not in request.files:
            message = "No file uploaded"
            return render_template("upload.html", message=message)

        myfile = request.files['file']
        fn = myfile.filename

        # Check if filename is empty
        if fn == '':
            message = "No file selected"
            return render_template("upload.html", message=message)

        # Validate file format
        accepted_formats = ['jpg', 'png', 'jpeg', 'jfif', 'JPG']
        if fn.split('.')[-1].lower() not in accepted_formats:
            message = "Image formats only accepted (jpg, png, jpeg, jfif)"
            return render_template("upload.html", message=message)

        # Save the uploaded file
        mypath = os.path.join('static/img', fn)
        myfile.save(mypath)

        try:
            # Load and preprocess the image
            input_tensor, original_image = load_image(mypath)

            # Predict the class
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            predicted_class_name = class_names[predicted_class]
            confidence = torch.softmax(output, dim=1)[0][predicted_class].item() * 100

            # Grad-CAM visualization
            target_layer = model.densenet.features[-1]
            grad_cam = GradCAM(model, target_layer)
            cam = grad_cam.generate_cam(input_tensor, predicted_class)

            # Convert input tensor to numpy image
            input_image = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
            input_image = np.uint8(255 * input_image)

            # Overlay Grad-CAM heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(input_image, 0.5, heatmap, 0.5, 0)

            # Save visualizations
            original_path = os.path.join('static/img', 'original_' + fn)
            heatmap_path = os.path.join('static/img', 'heatmap_' + fn)
            overlay_path = os.path.join('static/img', 'overlay_' + fn)

            plt.imsave(original_path, np.array(original_image))
            plt.imsave(heatmap_path, cam, cmap="jet")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Prepare prediction data
            prediction = {
                'class': predicted_class_name,
                'confidence': f"{confidence:.2f}%"
            }

            return render_template('upload.html',
                                 prediction=prediction,
                                 path=mypath,
                                 original_path=original_path,
                                 overlay_path=overlay_path)

        except Exception as e:
            message = f"Error processing image: {str(e)}"
            return render_template("upload.html", message=message)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug = True)