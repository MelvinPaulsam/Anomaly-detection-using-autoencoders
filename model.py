import torch.nn as nn
import torch.nn.functional as F
import torch
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import Dense


class AnomalyAE(nn.Module):
    def __init__(self,reg_lambda=0.01):
        super(AnomalyAE, self).__init__()
        self.reg_lambda = reg_lambda    

        self.conv1 = nn.Conv2d(1, 48, (11, 11), stride=(1, 1), padding=5)
        self.bn1 = nn.BatchNorm2d(48)

        self.conv2 = nn.Conv2d(48, 48, (9, 9), stride=(2, 2), padding=4)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 48, (7, 7), stride=(2, 2), padding=3)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48, 48, (5, 5), stride=(2, 2), padding=2)
        self.bn4 = nn.BatchNorm2d(48)

        self.conv5 = nn.Conv2d(48, 48, (3, 3), stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(48)
        self.conv6 = nn.Conv2d(48, 64, (3, 3), stride=(1, 1), padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv_tr1 = nn.ConvTranspose2d(
            48, 48, (5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.bn_tr1 = nn.BatchNorm2d(48)

        self.conv_tr2 = nn.ConvTranspose2d(
            96, 48, (7, 7), stride=(2, 2), padding=3, output_padding=1)
        self.bn_tr2 = nn.BatchNorm2d(48)

        self.conv_tr3 = nn.ConvTranspose2d(
            96, 48, (9, 9), stride=(2, 2), padding=4, output_padding=1)
        self.bn_tr3 = nn.BatchNorm2d(48)

        self.conv_tr4 = nn.ConvTranspose2d(
            96, 48, (11, 11), stride=(2, 2), padding=5, output_padding=1)
        self.bn_tr4 = nn.BatchNorm2d(48)

       # self.fc = nn.Linear(48 * 4 * 4, 256)  # FC layer added
       # self.fc_bn = nn.BatchNorm1d(256) #batch normalization
        self.conv_output = nn.Conv2d(96, 1, (1, 1), (1, 1))
        self.bn_output = nn.BatchNorm2d(1)

    #    self.encoder = Sequential([
    #                            Dense(64, activation='relu'),
    #                            Dense(32, activation='relu'),
     #                           Dense(16, activation='relu'),
     #                           Dense(8, activation='relu')
   # ])

    #    self.decoder = Sequential([
     #                          Dense(16, activation='relu'),
     #                          Dense(32, activation='relu'),
     #                          Dense(64, activation='relu'),
     #                          Dense(140, activation='sigmoid')
    #])
    
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.reg_lambda * l2_reg

    def forward(self, x):
        slope = 0.2
        x = F.leaky_relu((self.bn1(self.conv1(x))), slope)
        x1 = F.leaky_relu((self.bn2(self.conv2(x))), slope)
        x2 = F.leaky_relu((self.bn3(self.conv3(x1))), slope)
        x3 = F.leaky_relu((self.bn4(self.conv4(x2))), slope)
        x4 = F.leaky_relu((self.bn5(self.conv5(x3))), slope)

        x5 = F.leaky_relu(self.bn_tr1(self.conv_tr1(x4)), slope)
        x6 = F.leaky_relu(self.bn_tr2(
            self.conv_tr2(torch.cat([x5, x3], 1))), slope)
        x7 = F.leaky_relu(self.bn_tr3(
            self.conv_tr3(torch.cat([x6, x2], 1))), slope)
        x8 = F.leaky_relu(self.bn_tr4(
            self.conv_tr4(torch.cat([x7, x1], 1))), slope)
        x9 = F.leaky_relu(self.bn6(self.conv6(x8)), slope)

        #x_flat = torch.flatten(x4, 1)
        #x_fc = F.leaky_relu(self.fc(x_flat), slope) # Apply the FC layer

        # Reshape the output of the FC layer to match the shape of the deconvolutional layers
        #x_fc = x_fc.view(-1, 48, 4, 4)
        output = F.leaky_relu(self.bn_output(
            self.conv_output(torch.cat([x8, x], 1))), slope)
        return output

if __name__ == "__main__":
    x = torch.rand([16,1,512,512])
    model = AnomalyAE()
    y = model(x)
    regularization_loss = model.l2_regularization()
    total_loss = y + regularization_loss
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    print("Total loss with regularization: ", total_loss)


#model = Autoencoder()

# creating an early_stopping
#early_stopping = EarlyStopping(monitor='val_loss',
#                              patience = 2,
 #                              mode = 'min')
#
# Compiling the model
#model.compile(optimizer = 'adam',
#             loss = 'mae')
#
#CNN = Sequential(name="Sequential_CNN")
#
#CNN.add(Conv2D(16, kernel_size=(3, 3),
#               strides=(2, 2), padding="same",
#               activation="relu", input_shape=(28, 28, 1)))

#CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
#                     padding="valid"))

# Add another pair of Conv2D and MaxPooling2D for more model depth,
# followed by the flatten and multiple dense layers

#CNN.add(Conv2D(32, kernel_size=(3, 3),
#               strides=(2, 2), padding="same",
#               activation="relu"))

#CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
 #                    padding="valid"))

#CNN.add(Flatten())

#CNN.add(Dense(64, activation='relu'))
#CNN.add(Dense(32, activation='relu'))
#CNN.add(Dense(10, activation='softmax'))

#CNN.summary()
