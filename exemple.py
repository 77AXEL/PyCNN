from cnn import CNN

CNN = CNN()
#Initialize the model:
#CNN.init(
    image_size = 64, 
    batch_size = 32, 
    h1 = 128, 
    h2 = 64, 
    learning_rate = 0.001, 
    epochs = 400, 
    dataset_path = "data", 
    max_image=200
)

#Load the dataset folder for training:
#CNN.load_dataset()

#Train the model:
#CNN.train_model()

#Save the trainined model:
#CNN.save_model()

#Use a pretrained model:
#CNN.load_model("model.bin")

#Use the model:
#CNN.predict("exemple.jpg")
