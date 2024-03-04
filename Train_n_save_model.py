#Install the transformers library if you haven't already.
#pip install transformers


#Load the pre-trained model using AutoModelForSequenceClassification (or the appropriate model for your task)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
#I have used AutoModelForSequenceClassification above, u can choose it based on your task

# Step 1: Define your model name and directory to save the fine-tuned model
model_name = "distilbert-base-uncased"  #Name of the model you want to work upon for your task
save_directory = "./fine_tuned_model"   #In case you want to save it to local


# Step 2: Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification

# Step 3: Fine-tune the model on your dataset
# Define your training and validation data here, for example:
train_dataset = "train.csv"
eval_dataset = "eval.csv"

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    output_dir=save_directory,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
#Fine-tune the model on your dataset.

# Step 4: Save the fine-tuned model to a local directory
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

