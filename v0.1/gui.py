import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #encode the labels
import tkinter as tk #for GUI
from tkinter import ttk, messagebox#for GUI components and message boxes
import warnings#to ignore warnings
warnings.filterwarnings('ignore')

# ============ TRAIN THE MODEL ============
def train_model():
    """Train the Random Forest model on the crop dataset"""
    
    # Load data from the correct path
    try:
        df = pd.read_csv('dataset/Crop_recommendation.csv')
    except FileNotFoundError:
        messagebox.showerror("Error", "Dataset file not found!\nMake sure 'dataset/Crop_recommendation.csv' exists.")
        return None, None, None
    
    # Features and target
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y_encoded)
    
    # Calculate accuracy for info
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    accuracy = rf.score(X_test, y_test)
    
    return rf, le, accuracy

# ============ GUI APPLICATION ============
class CropRecommendationApp:
    def __init__(self, root, model, label_encoder, accuracy):
        self.root = root
        self.model = model
        self.label_encoder = label_encoder
        self.accuracy = accuracy
        
        self.root.title("🌾 Crop Recommendation System")
        self.root.geometry("500x650")
        self.root.resizable(False, False)
        
        # Configure style
        self.root.configure(bg='#f0f0f0')
        
        # Create main frame
        main_frame = tk.Frame(root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title = tk.Label(main_frame, text="🌱 Crop Recommendation System", 
                         font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2c5e2e')
        title.pack(pady=(0, 20))
        
        # Accuracy display
        acc_text = f"Model Accuracy: {accuracy*100:.2f}%"
        acc_label = tk.Label(main_frame, text=acc_text, font=('Arial', 10), 
                              bg='#f0f0f0', fg='#666')
        acc_label.pack(pady=(0, 20))
        
        # Create input fields
        self.entries = {}
        fields = [
            ('Nitrogen (N)', 'N', '0-140 kg/ha'),
            ('Phosphorus (P)', 'P', '0-100 kg/ha'),
            ('Potassium (K)', 'K', '0-200 kg/ha'),
            ('Temperature', 'temperature', '8-45 °C'),
            ('Humidity', 'humidity', '0-100 %'),
            ('pH', 'ph', '4-10'),
            ('Rainfall', 'rainfall', '0-300 mm')
        ]
        
        # Create a frame for inputs
        input_frame = tk.Frame(main_frame, bg='#f0f0f0')
        input_frame.pack(fill='both', expand=True)
        
        for i, (label, key, hint) in enumerate(fields):
            # Label
            lbl = tk.Label(input_frame, text=f"{label}:", font=('Arial', 11), 
                           bg='#f0f0f0', anchor='w', width=18)
            lbl.grid(row=i, column=0, padx=5, pady=8, sticky='w')
            
            # Entry
            entry = tk.Entry(input_frame, font=('Arial', 11), width=20)
            entry.grid(row=i, column=1, padx=5, pady=8)
            
            # Hint label
            hint_lbl = tk.Label(input_frame, text=hint, font=('Arial', 8), 
                                 bg='#f0f0f0', fg='#888')
            hint_lbl.grid(row=i, column=2, padx=5, pady=8, sticky='w')
            
            self.entries[key] = entry
        
        # Predict button
        self.predict_btn = tk.Button(main_frame, text="🌾 Predict Crop", 
                                      font=('Arial', 12, 'bold'),
                                      bg='#4CAF50', fg='white',
                                      command=self.predict,
                                      padx=20, pady=10,
                                      cursor='hand2')
        self.predict_btn.pack(pady=20)
        
        # Result frame
        result_frame = tk.Frame(main_frame, bg='white', relief='ridge', bd=2)
        result_frame.pack(fill='x', padx=10, pady=10)
        
        self.result_label = tk.Label(result_frame, text="🌿 Predicted Crop: --", 
                                      font=('Arial', 14, 'bold'),
                                      bg='white', fg='#2c5e2e',
                                      pady=15)
        self.result_label.pack()
        
        # Confidence frame (optional)
        confidence_frame = tk.Frame(main_frame, bg='#f0f0f0')
        confidence_frame.pack(fill='x', pady=10)
        
        self.confidence_label = tk.Label(confidence_frame, text="", 
                                          font=('Arial', 9),
                                          bg='#f0f0f0', fg='#666')
        self.confidence_label.pack()
        
        # Clear button
        self.clear_btn = tk.Button(main_frame, text="Clear All", 
                                    font=('Arial', 10),
                                    bg='#f44336', fg='white',
                                    command=self.clear_fields,
                                    padx=15, pady=5,
                                    cursor='hand2')
        self.clear_btn.pack(pady=5)
        
    def validate_input(self, value, field_name, min_val, max_val):
        """Validate user input"""
        try:
            num = float(value)
            if num < min_val or num > max_val:
                raise ValueError
            return True, num
        except ValueError:
            messagebox.showerror("Invalid Input", 
                                f"❌ {field_name} must be a number between {min_val} and {max_val}!\n"
                                f"Please enter a valid value.")
            return False, None
    
    def predict(self):
        """Get prediction from model"""
        
        # Define ranges for each feature
        ranges = {
            'N': (0, 140),
            'P': (0, 100),
            'K': (0, 200),
            'temperature': (8, 45),
            'humidity': (0, 100),
            'ph': (4, 10),
            'rainfall': (0, 300)
        }
        
        # Collect and validate inputs
        input_values = []
        features_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for feature in features_order:
            value = self.entries[feature].get().strip()
            
            # Check if empty
            if value == "":
                messagebox.showwarning("Missing Input", 
                                      f"⚠️ Please enter a value for {feature}!")
                return
            
            # Validate
            min_val, max_val = ranges[feature]
            field_name = feature.capitalize()
            if feature == 'temperature':
                field_name = 'Temperature (°C)'
            elif feature == 'humidity':
                field_name = 'Humidity (%)'
            elif feature == 'rainfall':
                field_name = 'Rainfall (mm)'
            elif feature == 'ph':
                field_name = 'pH'
            
            valid, num = self.validate_input(value, field_name, min_val, max_val)
            if not valid:
                return
            
            input_values.append(num)
        
        # Make prediction
        try:
            # Convert to numpy array and reshape for prediction
            input_array = np.array(input_values).reshape(1, -1)
            
            # Get prediction
            prediction_encoded = self.model.predict(input_array)[0]
            predicted_crop = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get prediction probabilities (confidence)
            probabilities = self.model.predict_proba(input_array)[0]
            confidence = probabilities[prediction_encoded] * 100
            
            # Display result
            self.result_label.config(text=f"🌾 Recommended Crop: {predicted_crop.upper()}")
            
            # Display confidence
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            
            # Color code confidence
            if confidence > 80:
                self.result_label.config(fg='#2c5e2e')
            elif confidence > 60:
                self.result_label.config(fg='#ff9800')
            else:
                self.result_label.config(fg='#f44336')
                
        except Exception as e:
            messagebox.showerror("Prediction Error", 
                                f"An error occurred during prediction:\n{str(e)}")
    
    def clear_fields(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(text="🌿 Predicted Crop: --", fg='#2c5e2e')
        self.confidence_label.config(text="")

# ============ MAIN EXECUTION ============
def main():
    print("🌾 Training Crop Recommendation Model...")
    
    # Train the model
    model, label_encoder, accuracy = train_model()
    
    if model is None:
        print("Failed to train model. Please check your dataset file.")
        return
    
    print(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
    print(f"📊 Number of crops: {len(label_encoder.classes_)}")
    print("🚀 Launching GUI...")
    
    # Create and run GUI
    root = tk.Tk()
    app = CropRecommendationApp(root, model, label_encoder, accuracy)
    root.mainloop()

if __name__ == "__main__":
    main()
