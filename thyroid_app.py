import tkinter as tk
from tkinter import Label, Entry, Button, messagebox, filedialog,Frame,Text
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json

# Load the model
model = tf.keras.models.load_model('thyroid_model.h5')

# Class names and recommendations
class_names = ['Hypothyroidism', 'Hyperthyroidism', 'Thyroid Nodules', 'Thyroiditis', 'Thyroid Cancer']
recommendations = {
    'Hypothyroidism': {'Diet': 'A hypothyroid-friendly diet includes iodine-rich foods (e.g., seaweed, fish), selenium sources (e.g., Brazil nuts), and zinc-rich foods (e.g., pumpkin seeds), while avoiding goitrogenic foods like raw cruciferous vegetables.  ', 'Medicine': 'Levothyroxine is the standard medication prescribed, a synthetic thyroid hormone that replaces deficient T4 levels and is taken on an empty stomach for best absorption.'},
    'Hyperthyroidism': {'Diet': 'A balanced diet with foods rich in antioxidants, calcium, and vitamin D is recommended, focusing on cruciferous vegetables (e.g., broccoli, kale), which may help slow thyroid hormone production. Avoid iodine-rich foods like seaweed and limit caffeine to manage symptoms.', 'Medicine': 'Common medications include *antithyroid drugs* like methimazole or propylthiouracil to reduce thyroid hormone production, beta-blockers like propranolol for symptom relief, and sometimes radioactive iodine therapy or surgery in severe cases.'},
    'Thyroid Nodules': {'Diet': 'A diet rich in iodine (like seafood and iodized salt) may help prevent iodine-deficient nodules, while selenium (found in nuts and seeds) supports thyroid health. Avoid excessive iodine intake and goitrogens (in raw cruciferous vegetables) for balance, depending on the underlying cause.', 'Medicine': 'Medications depend on the type; levothyroxine is used for hypothyroid nodules, while antithyroid drugs (like methimazole) manage hyperfunctioning ones. For inflammation-related nodules, doctors may prescribe steroids, and surgery or radioactive iodine is considered in severe cases.'},
    'Thyroiditis': {'Diet': 'A thyroid-friendly diet includes selenium-rich foods (nuts, fish), iodine in moderation, and anti-inflammatory foods like fruits and vegetables. Avoid processed foods, excess sugar, and gluten if sensitivity is suspected.', 'Medicine': 'Treatment may include levothyroxine for hypothyroidism, beta-blockers for hyperthyroid symptoms, and NSAIDs or steroids for inflammation. Autoimmune cases like Hashimoto may also require immune-modulating strategies.'},
    'Thyroid Cancer': {'Diet': 'A healthy diet rich in antioxidants, lean proteins, and fruits and vegetables is recommended to support the immune system and overall health. It is important to limit processed foods and excessive iodine intake, particularly if undergoing iodine therapy.', 'Medicine': 'Thyroid cancer is commonly treated with surgery, radioactive iodine therapy, and thyroid hormone replacement therapy. In some cases, targeted therapies or chemotherapy may be prescribed depending on the cancer type and stage.'}
}

# User authentication file
user_data_file = 'user_data.json'



# Thyroid knowledge base with 100 questions and answers
thyroid_qa = {
    "what is thyroid gland?": "The thyroid gland is a butterfly-shaped gland located in the front of the neck. It produces hormones that regulate the body's metabolism.",
    "where is the thyroid gland located in the body?": "The thyroid is located in the neck, just below the Adam's apple.",
    "what are the main functions of the thyroid gland?": "The thyroid gland regulates metabolism, energy generation, and body temperature. It also influences growth and development.",
    "what hormones does the thyroid produce?": "The thyroid produces two main hormones: thyroxine (T4) and triiodothyronine (T3).",
    "how does the thyroid gland affect metabolism?": "Thyroid hormones control the speed of metabolism. Too much hormone (hyperthyroidism) speeds up metabolism, while too little (hypothyroidism) slows it down.",
    "what is hypothyroidism?": "Hypothyroidism is a condition where the thyroid does not produce enough thyroid hormones. This can lead to symptoms like fatigue, weight gain, and depression.",
    "what is hyperthyroidism?": "Hyperthyroidism occurs when the thyroid produces too much thyroid hormone, causing symptoms such as weight loss, rapid heartbeat, and anxiety.",
    "what is thyroiditis?": "Thyroiditis is the inflammation of the thyroid gland, which can lead to changes in hormone levels, resulting in either hyperthyroidism or hypothyroidism.",
    "what are thyroid nodules?": "Thyroid nodules are abnormal growths or lumps in the thyroid gland. They are usually benign but can sometimes be cancerous.",
    "what is thyroid cancer?": "Thyroid cancer is a type of cancer that starts in the thyroid gland. It is relatively rare but treatable with surgery, radiation, and sometimes chemotherapy.",
    "what is Hashimoto's thyroiditis?": "Hashimoto's thyroiditis is an autoimmune disorder where the immune system attacks the thyroid, leading to hypothyroidism.",
    "what is Graves' disease?": "Graves' disease is an autoimmune disorder that causes hyperthyroidism. The body produces antibodies that stimulate the thyroid to produce too much hormone.",
    "what are the symptoms of hypothyroidism?": "Common symptoms of hypothyroidism include fatigue, weight gain, dry skin, depression, constipation, and sensitivity to cold.",
    "what are the symptoms of hyperthyroidism?": "Symptoms of hyperthyroidism include weight loss, rapid heartbeat, excessive sweating, nervousness, and difficulty sleeping.",
    "what are the symptoms of thyroid cancer?": "Thyroid cancer symptoms include a lump in the neck, difficulty swallowing, hoarseness, and pain in the neck or throat.",
    "what is a goiter?": "A goiter is an abnormal enlargement of the thyroid gland. It can be caused by iodine deficiency, thyroiditis, or thyroid cancer.",
    "how is hypothyroidism treated?": "Hypothyroidism is typically treated with synthetic thyroid hormone replacement, such as levothyroxine.",
    "how is hyperthyroidism treated?": "Hyperthyroidism can be treated with antithyroid medications, radioactive iodine therapy, or surgery to remove part of the thyroid.",
    "how is thyroid cancer treated?": "Thyroid cancer is treated with surgery, followed by radioactive iodine therapy to remove any remaining thyroid tissue or cancerous cells.",
    "can thyroid disorders be prevented?": "Thyroid disorders cannot always be prevented, but maintaining a healthy diet with sufficient iodine and regular check-ups can reduce risks.",
    "how can I maintain a healthy thyroid?": "To maintain a healthy thyroid, eat a balanced diet, avoid excessive iodine or lack of iodine, manage stress, and get regular checkups.",
    "what are the causes of hypothyroidism?": "Hypothyroidism can be caused by autoimmune diseases like Hashimoto's thyroiditis, iodine deficiency, or damage to the thyroid gland.",
    "what are the causes of hyperthyroidism?": "Hyperthyroidism can be caused by autoimmune diseases like Graves' disease, thyroiditis, or overconsumption of iodine.",
    "what is the difference between hypothyroidism and hyperthyroidism?": "Hypothyroidism occurs when the thyroid produces too little hormone, while hyperthyroidism occurs when the thyroid produces too much.",
    "what tests are used to diagnose thyroid disorders?": "Common tests include TSH (Thyroid Stimulating Hormone) test, T4 and T3 tests, and ultrasound for thyroid imaging.",
    "what is a thyroid ultrasound?": "A thyroid ultrasound is a non-invasive imaging test used to assess the thyroid gland's structure, detect nodules, or check for enlargement.",
    "what is the role of iodine in thyroid health?": "Iodine is essential for the production of thyroid hormones. A deficiency can lead to goiter and hypothyroidism.",
    "how can I prevent thyroid disorders?": "Ensure a balanced diet with sufficient iodine, avoid excessive iodine intake, and have regular thyroid check-ups.",
    "is thyroid disease genetic?": "Thyroid disease can run in families, particularly autoimmune thyroid diseases like Hashimoto's thyroiditis and Graves' disease.",
    "can stress affect thyroid function?": "Chronic stress may influence thyroid function, potentially leading to thyroid disorders such as hypothyroidism or hyperthyroidism.",
    "can pregnancy affect thyroid function?": "Pregnancy can cause changes in thyroid function. Hypothyroidism or hyperthyroidism during pregnancy can affect both the mother and baby.",
    "what is a thyroid panel?": "A thyroid panel is a set of blood tests that measure the levels of thyroid hormones (T3, T4) and TSH to assess thyroid function.",
    "what is TSH?": "TSH (Thyroid Stimulating Hormone) is a hormone produced by the pituitary gland that stimulates the thyroid to produce T3 and T4.",
    "what is the normal range for TSH levels?": "The normal range for TSH levels is typically between 0.4 and 4.0 mIU/L, but this can vary depending on the lab and the individual.",
    "what are thyroid antibodies?": "Thyroid antibodies are proteins produced by the immune system that attack the thyroid in autoimmune diseases like Hashimoto's and Graves' disease.",
    "what is the role of selenium in thyroid health?": "Selenium is an essential trace mineral that plays a role in thyroid hormone metabolism and helps protect the thyroid from oxidative damage.",
    "what is the connection between thyroid and cholesterol?": "Both hypothyroidism and hyperthyroidism can affect cholesterol levels, with hypothyroidism often leading to high cholesterol levels.",
    "what is the impact of thyroid disorders on weight?": "Hypothyroidism can lead to weight gain, while hyperthyroidism may cause weight loss due to an increased metabolism.",
    "can thyroid problems cause hair loss?": "Thyroid disorders, particularly hypothyroidism, can cause hair thinning or hair loss.",
    "what is the treatment for thyroid nodules?": "Treatment for thyroid nodules may include observation, biopsy, or surgery, depending on the size and type of nodule.",
    "can iodine deficiency lead to thyroid problems?": "Yes, iodine deficiency can cause goiter and hypothyroidism, as iodine is essential for the production of thyroid hormones.",
    "what is radioactive iodine therapy?": "Radioactive iodine therapy is used to treat hyperthyroidism or thyroid cancer. It destroys thyroid tissue, reducing hormone production.",
    "what is the thyroid-stimulating hormone (TSH) test?": "The TSH test measures the level of TSH in the blood to help diagnose thyroid dysfunction. High TSH levels typically indicate hypothyroidism.",
    "how does age affect thyroid function?": "Thyroid function can change with age. Older adults may have a higher risk of hypothyroidism or other thyroid conditions.",
    "what are the complications of untreated hypothyroidism?": "Untreated hypothyroidism can lead to heart disease, infertility, nerve damage, and mental health issues like depression.",
    "what are the complications of untreated hyperthyroidism?": "Untreated hyperthyroidism can lead to heart problems, osteoporosis, and a potentially life-threatening condition called thyroid storm.",
    "what is a thyroidectomy?": "A thyroidectomy is a surgical procedure in which part or all of the thyroid gland is removed, often used to treat thyroid cancer or hyperthyroidism.",
    "how does obesity affect thyroid function?": "Obesity can influence thyroid function, and individuals with thyroid dysfunction may be more likely to experience weight issues.",
    "what are the risk factors for thyroid cancer?": "Risk factors for thyroid cancer include gender (women are more likely), age, radiation exposure, and family history.",
    "can thyroid disorders cause infertility?": "Yes, thyroid disorders, particularly hypothyroidism and hyperthyroidism, can affect fertility in both men and women.",
    "what is subclinical hypothyroidism?": "Subclinical hypothyroidism is a mild form of hypothyroidism where the TSH is slightly elevated, but T3 and T4 levels are normal.",
    "how is subclinical hypothyroidism treated?": "Subclinical hypothyroidism may not always require treatment, but if symptoms are present or TSH levels continue to rise, thyroid hormone replacement therapy may be considered.",
    "what is a thyroid scan?": "A thyroid scan is a nuclear medicine test that evaluates the thyroid gland's function and structure by tracking radioactive material injected into the body.",
    "what are the types of thyroid cancer?": "There are four main types of thyroid cancer: papillary, follicular, medullary, and anaplastic thyroid cancer.",
    "what is medullary thyroid cancer?": "Medullary thyroid cancer is a rare form of thyroid cancer that originates in the C cells of the thyroid and can spread to other parts of the body.",
    "what is anaplastic thyroid cancer?": "Anaplastic thyroid cancer is a rare and aggressive form of thyroid cancer that tends to grow rapidly and is difficult to treat.",
    "what is the function of the parathyroid glands?": "The parathyroid glands are small glands located near the thyroid that regulate calcium levels in the blood.",
    "how does the thyroid affect the skin?": "Thyroid disorders can cause changes in the skin, such as dryness in hypothyroidism or thinning skin in hyperthyroidism.",
    "what is the role of vitamin D in thyroid health?": "Vitamin D plays a role in thyroid hormone production and immune function, and deficiency has been linked to autoimmune thyroid disease.",
    "what is the relationship between thyroid and heart disease?": "Thyroid dysfunction, particularly hypothyroidism, can lead to high cholesterol and other cardiovascular issues, while hyperthyroidism can cause an increased risk of arrhythmias.",
    "can thyroid problems cause mood changes?": "Yes, thyroid problems can cause mood changes. Hypothyroidism may lead to depression, while hyperthyroidism can cause anxiety or irritability.",
    "what are the treatment options for thyroid cancer?": "Treatment options for thyroid cancer include surgery, radioactive iodine therapy, external radiation therapy, and sometimes chemotherapy.",
    "how long does it take for thyroid medication to work?": "It typically takes several weeks to months for thyroid hormone medication to reach its full effect and balance thyroid hormone levels.",
    "can diet affect thyroid function?": "Yes, a diet that is deficient in iodine or other nutrients can affect thyroid function. Balanced nutrition supports thyroid health.",
    "what is the impact of thyroid disorders on the menstrual cycle?": "Thyroid disorders, particularly hypothyroidism and hyperthyroidism, can cause menstrual irregularities, such as heavy periods or missed periods.",
    "what is the connection between thyroid and liver function?": "Thyroid disorders can affect liver function, and liver problems may influence thyroid hormone metabolism and clearance.",
    "how is thyroid disease managed in children?": "Thyroid disease in children is managed through hormone replacement therapy for hypothyroidism or antithyroid medication for hyperthyroidism.",
    "what is the relationship between thyroid and blood sugar?": "Thyroid dysfunction can influence blood sugar levels, with hypothyroidism potentially causing insulin resistance and hyperthyroidism leading to low blood sugar.",
    "what is the role of T4 in the body?": "T4 (thyroxine) is a thyroid hormone that is converted into the more active T3 form in the body. It regulates metabolism, growth, and development.",
    "can thyroid disorders affect sleep?": "Yes, thyroid disorders, particularly hyperthyroidism, can cause sleep disturbances, while hypothyroidism may lead to fatigue and excessive sleepiness.",
    "what is the role of T3 in the body?": "T3 (triiodothyronine) is the active form of thyroid hormone that helps regulate the body's metabolism, energy production, and temperature.",
    "how can I improve thyroid health naturally?": "Improving thyroid health involves maintaining a balanced diet, avoiding stress, getting regular exercise, and ensuring proper iodine intake.",
    "what are the symptoms of thyroid problems in men?": "In men, thyroid problems can lead to symptoms such as fatigue, weight gain, hair loss, low libido, and depression.",
    "what are the effects of untreated hyperthyroidism?": "Untreated hyperthyroidism can lead to heart disease, osteoporosis, and thyroid storm, a life-threatening condition.",
    "what is the best diet for someone with hypothyroidism?": "A diet rich in fruits, vegetables, lean protein, and whole grains is beneficial for hypothyroidism, with a focus on iodine, selenium, and vitamin D.",
    "what are the best foods for thyroid health?": "Foods like seaweed, eggs, fish, nuts, and dairy products, which are rich in iodine, selenium, and zinc, are good for thyroid health.",
}

# Continue using the same tkinter application from the previous code


class ThyroidApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thyroid Diagnosis System")
        self.current_user = None

        # Styling colors
        self.bg_color = "#f0f8ff"
        self.btn_color = "#4682b4"
        self.logout_btn_color = "#b22222"
        self.font_large = ("Helvetica", 18, "bold")
        self.font_normal = ("Helvetica", 14)

        # Navigation and content frames
        self.nav_frame = tk.Frame(self.root, bg=self.bg_color)
        self.nav_frame.pack(fill="x")

        self.content_frame = tk.Frame(self.root, bg=self.bg_color)
        self.content_frame.pack(fill="both", expand=True)

        # Navigation Buttons
        self.logout_button = Button(self.nav_frame, text="Logout", command=self.logout, bg=self.logout_btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12)
        self.logout_button.pack(side="left", padx=10, pady=10)
        self.logout_button.pack_forget()

        self.login_register_button = Button(self.nav_frame, text="Login/Register", command=self.show_login_page, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12)
        self.login_register_button.pack(side="left", padx=10, pady=10)

        self.home_button = Button(self.nav_frame, text="Home", command=self.show_home_page, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12)
        self.predict_button = Button(self.nav_frame, text="Predict", command=self.show_predict_page, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12)
        self.chatbot_button = Button(self.nav_frame, text="Chatbot", command=self.show_chatbot_page, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12)

        self.show_login_page()

    def show_home_page(self):
        self.clear_content_frame()
    

    # Welcome message with increased font size and custom color
        Label(
            self.content_frame, 
            text="Welcome to the Thyroid Diagnosis System!", 
            font=("Helvetica", 24, "bold"),  # Increased font size
            bg=self.bg_color, 
            fg="#4B0082"  # Indigo color for the welcome text
        ).pack(pady=20)

    # New sentence for awareness with a different color
        Label(
            self.content_frame, 
            text="Early diagnosis saves lives. Trust our deep learning-based thyroid diagnosis system to be your companion on the journey to better health.", 
            wraplength=600, 
            justify="center", 
            font=self.font_normal, 
            bg=self.bg_color, 
            fg="#2E8B57"  # Sea Green color for the awareness sentence
        ).pack(pady=10)

        # Add an image below the description
        try:
        # Load and display the image
            img = Image.open('image8.png')  # Update the file path if needed
            img_resized = img.resize((800, 400))  # Resize the image as necessary
            img_tk = ImageTk.PhotoImage(img_resized)
        
            Label(self.content_frame, image=img_tk, bg=self.bg_color).pack(pady=5)
            self.content_frame.image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            print(f"Error loading image: {e}")
            Label(
                self.content_frame, 
                text="Image could not be loaded.", 
                font=self.font_normal, 
                bg=self.bg_color, 
                fg="red"
            ).pack(pady=10)

    # Detailed description about the platform
        Label(
            self.content_frame, 
            text="Our platform helps diagnose thyroid conditions using Convolutional Neural Networks technique, providing accurate predictions and personalized recommendations. It provides dietary and medicinal guidance while featuring a chatbot designed to deliver clear and accurate answers to all thyroid-related questions.", 
            wraplength=600, 
            justify="center", 
            font=self.font_normal, 
            bg=self.bg_color
        ).pack(pady=20)

    


    def show_login_page(self):
        self.clear_content_frame()
        Label(self.content_frame, text="Thyroid Diagnosis System", font=("Arial", 24, "bold"), bg=self.bg_color, fg="#008000").pack(pady=20)  # Modify the color as needed

            
    # Add an image below the description
        try:
        # Load and display the image
            img = Image.open('image2.jpg')  # Update the file path if needed
            img_resized = img.resize((100, 100))  # Resize the image as necessary
            img_tk = ImageTk.PhotoImage(img_resized)
        
            Label(self.content_frame, image=img_tk, bg=self.bg_color).pack(pady=5)
            self.content_frame.image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            print(f"Error loading image: {e}")
            Label(
                self.content_frame, 
                text="Image could not be loaded.", 
                font=("Arial", 14), 
                bg=self.bg_color, 
                fg="red"
            ).pack(pady=10)
        Label(self.content_frame, text="Login", font=self.font_large, bg=self.bg_color).pack(pady=20)

    # Label and entry for username
        Label(self.content_frame, text="Username:", bg=self.bg_color, font=self.font_normal).pack()
        username_entry = Entry(self.content_frame, font=self.font_normal)
        username_entry.pack()

    # Label and entry for password
        Label(self.content_frame, text="Password:", bg=self.bg_color, font=self.font_normal).pack()
        password_entry = Entry(self.content_frame, show="*", font=self.font_normal)
        password_entry.pack()

    # Function to handle login process
        def login():
            username = username_entry.get()
            password = password_entry.get()
        
            if not username or not password:
                messagebox.showerror("Error", "Please enter a username and password.")
                return

            try:
                with open(user_data_file, 'r') as f:
                    user_data = json.load(f)
            except FileNotFoundError:
                user_data = {}

            if username in user_data:
                if user_data[username] == password:
                    messagebox.showinfo("Success", f"Welcome back, {username}!")
                    self.current_user = username
                    self.setup_navigation()
                    self.show_home_page()
                else:
                    messagebox.showerror("Error", "Incorrect password.")
            else:
                messagebox.showerror("Error", "Account not found. Please register first.")

    # Button to login
        Button(self.content_frame, text="Login", command=login, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12).pack(pady=10)

    # Link to switch to the registration page
        def show_register_page():
            self.show_register_page()

        Button(self.content_frame, text="Create an account", command=show_register_page, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12).pack(pady=10)

    def show_register_page(self):
        self.clear_content_frame()
        Label(self.content_frame, text="Register", font=self.font_large, bg=self.bg_color).pack(pady=10)

    # Label and entry for username
        Label(self.content_frame, text="Username:", bg=self.bg_color, font=self.font_normal).pack()
        username_entry = Entry(self.content_frame, font=self.font_normal)
        username_entry.pack()

    # Label and entry for password
        Label(self.content_frame, text="Password:", bg=self.bg_color, font=self.font_normal).pack()
        password_entry = Entry(self.content_frame, show="*", font=self.font_normal)
        password_entry.pack()

    # Function to handle registration process
        def register():
            username = username_entry.get()
            password = password_entry.get()

            if not username or not password:
                messagebox.showerror("Error", "Please enter a username and password.")
                return

            try:
                with open(user_data_file, 'r') as f:
                    user_data = json.load(f)
            except FileNotFoundError:
                user_data = {}

            if username in user_data:
                messagebox.showerror("Error", f"Account already created for {username}. Please login.")
            else:
                user_data[username] = password
                with open(user_data_file, 'w') as f:
                    json.dump(user_data, f)
                messagebox.showinfo("Success", f"Account created for {username}!")
                self.current_user = username
                self.setup_navigation()
                self.show_home_page()

    # Button to register
        Button(self.content_frame, text="Register", command=register, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12).pack(pady=10)

    # Link to switch to the login page
        def show_login_page():
            self.show_login_page()

        Button(self.content_frame, text="Already have an account? Login", command=show_login_page, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12).pack(pady=10)

    def logout(self):
        self.current_user = None
        self.setup_navigation()
        self.show_login_page()

    def setup_navigation(self):
        if self.current_user:
            self.login_register_button.pack_forget()
            self.logout_button.pack(side="left", padx=10, pady=10)
            self.home_button.pack(side="left", padx=10, pady=10)
            self.predict_button.pack(side="left", padx=10, pady=10)
            self.chatbot_button.pack(side="left", padx=10, pady=10)
        else:
            self.logout_button.pack_forget()
            self.login_register_button.pack(side="left", padx=10, pady=10)
            self.home_button.pack_forget()
            self.predict_button.pack_forget()
            self.chatbot_button.pack_forget()

    def show_predict_page(self):
        self.clear_content_frame()

        # Label for instructions
        Label(self.content_frame, text="Upload an image for thyroid diagnosis", font=self.font_large, bg=self.bg_color).pack(pady=10)

        # Function to upload and predict the image
        def upload_and_predict():
    # Ask user to select an image file
            file_path = filedialog.askopenfilename()
            if file_path:
                try:
            # Open and display the image using PIL
                    img = Image.open(file_path)
                    img_resized = img.resize((224, 224))  # Resize the image to the model's expected input size
                    img_tk = ImageTk.PhotoImage(img_resized)

            # Display the image in the tkinter window
                    img_label.config(image=img_tk)
                    img_label.image = img_tk  # Keep a reference to avoid garbage collection

            # Preprocess the image for prediction
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

            # Make prediction
                    prediction = model.predict(img_array)[0]  # Get prediction probabilities
                    max_confidence = np.max(prediction) * 100  # Convert to percentage
                    predicted_index = np.argmax(prediction)  # Get the index of the highest confidence score

            # Check if confidence is below the threshold
                    confidence_threshold = 70.0
                    if max_confidence < confidence_threshold:
                # If confidence is too low, display a warning message
                        result_label.config(
                            text="It is not a thyroid image.\n"
                                "Please upload a valid thyroid-related image."
                        )
                    else:
                # If confidence is sufficient, display the prediction and recommendations
                        predicted_class = class_names[predicted_index]
                        rec = recommendations[predicted_class]

                # Update result labels with prediction
                        result_label.config(
                            text="This person is suffering from Thyroid.\n"
                                f"Prediction: {predicted_class}\n"
                                 f"Confidence: {max_confidence:.2f}%\n"
                                 f"Diet: {rec['Diet']}\n"
                                 f"Medicine: {rec['Medicine']}"
                        )
                except Exception as e:
            # Display any error message during the process
                    result_label.config(
                        text=f"Error processing the image: {e}"
                    )

# Button to upload the image
        Button(self.content_frame, text="Upload Image", command=upload_and_predict, 
            bg=self.btn_color, fg="white", font=self.font_normal, 
            relief="flat", borderwidth=0, padx=30, pady=12).pack(pady=10)

# Label to display the uploaded image
        img_label = Label(self.content_frame, bg=self.bg_color)
        img_label.pack(pady=10)

        result_label = Label(
        self.content_frame,
        bg=self.bg_color,
        font=self.font_normal,
        wraplength=1000 ,
        justify="center", # Adjust this value based on your window width
        )
        result_label.pack(pady=10)


    def show_chatbot_page(self):
        self.clear_content_frame()

    # Create a frame to hold the image and chatbot content
        chatbot_frame = Frame(self.content_frame, bg=self.bg_color)
        chatbot_frame.pack(pady=20)

    # Load the image
        try:
            img = Image.open('image7.png')  # Path to the uploaded image
            img_resized = img.resize((400, 400))  # Resize the image to fit the layout
            img_tk = ImageTk.PhotoImage(img_resized)

        # Add the image to the left side
            image_label = Label(chatbot_frame, image=img_tk, bg=self.bg_color)
            image_label.grid(row=0, column=0, padx=20, pady=10)  # Place the image in the first column
            chatbot_frame.image = img_tk  # Keep reference to avoid garbage collection

        except Exception as e:
            print(f"Error loading image: {e}")
            Label(chatbot_frame, text="Image could not be loaded.", font=self.font_normal, bg=self.bg_color, fg="red").grid(row=0, column=0)

    # Add the chatbot content in the second column of the frame
        chat_content = Frame(chatbot_frame, bg=self.bg_color)
        chat_content.grid(row=0, column=1, padx=20, pady=10)

    # Label for Chatbot
        Label(chat_content, text="Thyroid Chatbot", font=self.font_large, bg=self.bg_color).pack(pady=20)

    # Chat history display area
        chat_history = Text(chat_content, height=15, width=50, font=self.font_normal, state="disabled", wrap="word", bg=self.bg_color)
        chat_history.pack(pady=10)

    # Entry field for user input
        user_input = Entry(chat_content, font=self.font_normal, width=40)
        user_input.pack(pady=10)

    # Function to handle user query and show response
        def ask_chatbot():
            user_query = user_input.get()
            if user_query:
                chat_history.config(state="normal")
                chat_history.insert(tk.END, f"You: {user_query}\n")
                chat_history.insert(tk.END, f"Bot: {self.get_chatbot_response(user_query)}\n")
                chat_history.config(state="disabled")
                user_input.delete(0, tk.END)

    # Button to submit query
        Button(chat_content, text="Ask", command=ask_chatbot, bg=self.btn_color, fg="white", font=self.font_normal, relief="flat", borderwidth=0, padx=30, pady=12).pack(pady=10)


    def get_chatbot_response(self, query):
        return thyroid_qa.get(query, "Sorry, I don't understand the question.")

    def clear_content_frame(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ThyroidApp(root)
    root.mainloop()


