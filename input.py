from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('models/artgan.keras')

# Load and preprocess the image
img_path = input('Enter the path to the image: ')
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0


# Make predictions
predictions = model.predict(img_array)

# Class labels
class_labels = [
  'Abstract_Expressionism', 'Cubism', 'Minimalism', 'Realism',
  'Action_painting', 'Early_Renaissance', 'Naive_Art_Primitivism', 'Rococo',
  'Analytical_Cubism', 'Expressionism', 'New_Realism', 'Romanticism',
  'Art_Nouveau_Modern', 'Fauvism', 'Northern_Renaissance', 'Symbolism',
  'Baroque', 'High_Renaissance', 'Pointillism', 'Synthetic_Cubism',
  'Color_Field_Painting', 'Impressionism', 'Pop_Art', 'Ukiyo_e',
  'Contemporary_Realism', 'Mannerism_Late_Renaissance', 'Post_Impressionism'
]

# Get predicted genre
predicted_class_index = np.argmax(predictions[0])
predicted_genre = class_labels[predicted_class_index]

# Print the predicted genre
print(f'The predicted genre is: {predicted_genre}')

# Optional: Print all probabilities
for i, genre in enumerate(class_labels):
    print(f'{genre}: {predictions[0][i] * 100:.2f}%')
