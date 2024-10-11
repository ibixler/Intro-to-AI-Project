from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = load_model('models/artgan.keras')



def process_paint(img_path):
  img = image.load_img(img_path, target_size=(250, 250))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0
  
  predictions = model.predict(img_array)

  class_labels = [
    'Abstract_Expressionism', 'Cubism', 'Minimalism', 'Realism',
    'Action_painting', 'Early_Renaissance', 'Naive_Art_Primitivism', 'Rococo',
    'Analytical_Cubism', 'Expressionism', 'New_Realism', 'Romanticism',
    'Art_Nouveau_Modern', 'Fauvism', 'Northern_Renaissance', 'Symbolism',
    'Baroque', 'High_Renaissance', 'Pointillism', 'Synthetic_Cubism',
    'Color_Field_Painting', 'Impressionism', 'Pop_Art', 'Ukiyo_e',
    'Contemporary_Realism', 'Mannerism_Late_Renaissance', 'Post_Impressionism'
  ]

  predicted_class_index = np.argmax(predictions[0])
  predicted_genre = class_labels[predicted_class_index]

  ret = (f'The predicted genre is: {predicted_genre}\n')
  for i, genre in enumerate(class_labels):
      ret += (f'\n{genre}: {predictions[0][i] * 100:.2f}%\n')
  return ret

