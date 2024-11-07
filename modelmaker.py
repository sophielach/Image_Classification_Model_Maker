from icrawler.builtin import BingImageCrawler
from huggingface_hub import create_repo
from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
from torchvision.transforms import RandomAffine, Compose, Normalize, ToTensor, Resize, 
import numpy as np
import os
from delete_images import *
import shutil

class ModelMaker:
	def __init__(self, keywords, num_images, key, dataset_name, model_name, train_epochs):
		self.keywords = keywords
		self.num_images = num_images # Now a list corresponding to each keyword
		self.key = key
		self.dataset_name = dataset_name
		self.model_name = model_name
		self.train_epochs = train_epochs

		self.dataset_id = ''

	def download_images(self):
		# Create folder for all images
		image_path = './images/'
		if not os.path.exists(image_path):
			os.makedirs(image_path)

		# Go through each keyword and corresponding number of images
		for keyword, num_images_for_keyword in zip(self.keywords, self.num_images):
			# Create a folder for each keyword
			save_path = image_path + keyword
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			# Download images from Bing
			bing_crawler = BingImageCrawler(storage={'root_dir': save_path})
			bing_crawler.crawl(keyword=keyword, max_num=num_images_for_keyword)

	def filter_images(self):
		# open Gradio interface to delete photos as necessary
		data_dir = './images'
		folders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    
		# Open the Gradio interface for each folder
		for folder in folders:
			#category_name = os.path.basename(folder)
			print(f"Opening interface for catgory: {folder}") # change this to remove .images
			interface = create_interface(folder)
			interface.launch()

	def merge_keywords(self, list_of_keys):
		# merge all keywords' images into the first one of the list
		# the first keyword may be 'empty' or already contain images
		if not list_of_keys:
			print("No keywords provided for merging.")
			return

		base_dir = './images'
		target_folder = os.path.join(base_dir, list_of_keys[0])
		
		# Ensure the target folder exists - if not, make one
		if not os.path.exists(target_folder):
			os.makedirs(target_folder)

		# Loop over the other keywords and move their contents
		for keyword in list_of_keys[1:]:
			source_folder = os.path.join(base_dir, keyword)
			
			# Check if the source folder exists
			if os.path.exists(source_folder):
				# Move each file from the source folder to the target folder
				for filename in os.listdir(source_folder):
					source_path = os.path.join(source_folder, filename)
					target_path = os.path.join(target_folder, filename)

					# If a file with the same name exists, rename the file
					if os.path.exists(target_path):
						base, ext = os.path.splitext(filename)
						counter = 1
						while os.path.exists(target_path):
							target_path = os.path.join(target_folder, f"{base}_{counter}{ext}")
							counter += 1

					shutil.move(source_path, target_path)
				
				# Remove the now-empty source folder
				os.rmdir(source_folder)
				print(f"Merged and removed folder: {source_folder}")
			else:
				print(f"Source folder does not exist: {source_folder}")

		print(f"All images from {list_of_keys[1:]} merged into {target_folder}.")

	def upload_dataset(self):
		# Create a dataset repo
		self.dataset_id = create_repo(self.dataset_name, token=self.key, repo_type="dataset").repo_id

		# Load images
		dataset = load_dataset('imagefolder', data_dir='./images',  split='train')

		# Push dataset to Huggingface
		dataset.push_to_hub(self.dataset_id, token=self.key)

		print('Dataset was uploaded to:', self.dataset_id)

	def train_model(self):
		# Load dataset
		ds = load_dataset(self.dataset_id, split="train")

		# Split data between test and train
		ds = ds.train_test_split(test_size=0.2)

		# Create a dictionary that maps the label name to an integer and vice versa
		labels = ds["train"].features["label"].names
		label2id, id2label = dict(), dict()
		for i, label in enumerate(labels):
			label2id[label] = str(i)
			id2label[str(i)] = label

		# Preprocess: load a ViT image processor to process the image into a tensor
		checkpoint = "google/vit-base-patch16-224-in21k"
		image_processor = AutoImageProcessor.from_pretrained(checkpoint)

		# removed random crop. 

		# Apply some image transformations to the images to make the model more robust against overfitting
		# Crop a random part of the image, resize it, and normalize it with the image mean and standard deviation
		normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
		size = (
			image_processor.size["shortest_edge"]
			if "shortest_edge" in image_processor.size
			else (image_processor.size["height"], image_processor.size["width"])
		)
		_transforms = Compose([
			Resize(size),                 # Resize to match model input size
			RandomAffine(
				degrees=5,                # Small rotation for slight variability
				translate=(0.05, 0.05),   # Minor translation to avoid extreme displacement
				scale=(0.95, 1.05)        # Minor scaling to keep items realistically sized
			),
			ToTensor(),
			normalize
		])
		# Preprocessing function to apply the transforms and return the pixel_values 
		def transforms(examples):
			examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
			del examples["image"]
			return examples

		# Apply the preprocessing function over the entire dataset
		ds = ds.with_transform(transforms)

		# Create a batch of examples using DefaultDataCollator.
		data_collator = DefaultDataCollator()

		# Load an evaluation method 
		accuracy = evaluate.load("accuracy")

		# Function that passes your predictions and labels to compute to calculate the accuracy:
		def compute_metrics(eval_pred):
			predictions, labels = eval_pred
			predictions = np.argmax(predictions, axis=1)
			return accuracy.compute(predictions=predictions, references=labels)

		#  Load ViT with AutoModelForImageClassification, number of labels, and the label mappings
		model = AutoModelForImageClassification.from_pretrained(
			checkpoint,
			num_labels=len(labels),
			id2label=id2label,
			label2id=label2id,
		)

		# Define training hyperparameters
		training_args = TrainingArguments(
			output_dir=self.model_name,
			remove_unused_columns=False,
			evaluation_strategy="epoch",
			save_strategy="epoch",
			learning_rate=5e-5,
			per_device_train_batch_size=16,
			gradient_accumulation_steps=4,
			per_device_eval_batch_size=16,
			num_train_epochs=self.train_epochs,
			warmup_ratio=0.1,
			logging_steps=10,
			load_best_model_at_end=True,
			metric_for_best_model="accuracy",
			push_to_hub=True,
			resume_from_checkpoint=True,
			hub_token=self.key
		)

		# Pass the training arguments to Trainer
		trainer = Trainer(
			model=model,
			args=training_args,
			data_collator=data_collator,
			train_dataset=ds["train"],
			eval_dataset=ds["test"],
			tokenizer=image_processor,
			compute_metrics=compute_metrics
		)

		# Begin training
		trainer.train()

		# Push model to hub
		trainer.push_to_hub()
