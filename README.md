# Easy Image Classification Model Maker
Create a custom image classification model with a few lines of code. This module scrapes images, formats and uploads the image dataset to 🤗, and trains a 🤗 model. Built on top of 🤗 Transformers and 🤗 Datasets.


Added features from forked version: 
1. Download varying numbers of pictures for each category
2. Use a Gradio interface to sort through pictures and manually delete irrelevant ones before uploading dataset, via `filter_images()`
3. Merge categories before uploading the dataset via `merge_categories`, which takes a list of all categories and places all images under the category of the first one in the list
4. Preprocess images with affine instead of cropping
 

 Future additions I will likely add:
5. Remove duplicate photos from each category with `remove_duplicate_images`
6. Allow the option of affine **or** cropping for data preprocessing
7. `load_dataset`: load a dataset from huggingface and add it seamlessly into './images'
8. `visualize_misclasses`: see which images the model most struggle with, and if they should be reasigned/corrected
9. `retrain`: add the ability to retrain the model again after shifting aronud the dataset. allow for more hyperparameter tuning
10. demo notebook
11. update `README` to reflect the new functions and their parameters


## Installation
```bash
pip install -r requirements.txt
```

## Train Model
Import the module
```python
import modelmaker
```
Define the model and dataset parameters:
- **keyword** list of strings will be the labels of the model
- **num_images** number of images in the training dataset
- **key** HuggingFace write access token can be created [here](https://huggingface.co/settings/tokens).
- **dataset_name** name of dataset that will uploaded to HuggingFace
- **model_name** name of model that will be uploaded to HuggingFace
- **train_epochs** number of training epochs the model will go through
```python
model = modelmaker.ModelMaker(keywords = ['cubism', 'impressionism', 'abstract expressionism'],
                              num_images = 100,
                              key = 'YOUR_TOKEN',
                              dataset_name = 'art_dataset',
                              model_name = 'art_classifier',
                              train_epochs = 10)
```
Download images from Bing into the './images' folder. It is suggested to manually go through the image folders to make sure there isn't any incorrect images in their respective folders. 
```python
model.download_images()
```
Upload dataset to HuggingFace
```python
model.upload_dataset()
```
Train the model and upload it to HuggingFace
```python
model.train_model()
```

## Model Usage
### Inference API Widget
Go to the model page, which can be found on your HuggingFace page. Drag and drag images onto the Inference API section to test it.

### Python
```python
from transformers import pipeline

pipe = pipeline("image-classification", model="tonyassi/art_classifier")
result = pipe('image.png')

print(result)
```

### JavaScript API
```js
async function query(filename) {
	const data = fs.readFileSync(filename);
	const response = await fetch(
		"https://api-inference.huggingface.co/models/tonyassi/art_classifier",
		{
			headers: { Authorization: "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" },
			method: "POST",
			body: data,
		}
	);
	const result = await response.json();
	return result;
}

query("art.jpg").then((response) => {
	console.log(JSON.stringify(response));
});
```
