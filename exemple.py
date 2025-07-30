from cnn import CNN

CNN = CNN()
CNN.init(
    image_size = 64, 
    batch_size = 32, 
    h1 = 128, 
    h2 = 64, 
    learning_rate = 0.001, 
    epochs = 400, 
    dataset_path = "data", 
    max_image=200
)

CNN.load_model("model.bin")

CNN.predict("exemple.jpg")