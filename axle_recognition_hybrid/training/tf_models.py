from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import applications
import tensorflow as tf
from dark_attention import DarkAttention, MaskLayer

def VGG16(class_count):
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def VGG19(class_count):
    model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def DenseNet121(class_count):
    model = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def MobileNetV3Small(class_count):
    model = applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def ResNet101V2(class_count):
    model = applications.ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def img_channel_scale(t):
    return tf.cast(t, tf.float32) / 255.0

def conv_block(x, filters, convs=2, name=None):
    for i in range(convs):
        x = Conv2D(filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name=None if name is None else f"{name}_conv{i+1}")(x)
    return x


def custom_19(class_count):
    img = Input(shape=(224, 224, 3), name='input_layer')
    x = Lambda(img_channel_scale)(img)
    
    x = conv_block(x, 64, convs=2, name="block1")
    x = MaxPool2D((2,2), strides=2, name='pool1')(x)

    x = conv_block(x, 128, convs=2, name="block2")
    x = MaxPool2D((2,2), strides=2, name='pool2')(x)

    x = conv_block(x, 256, convs=4, name="block3")
    x = MaxPool2D((2,2), strides=2, name='pool3')(x)

    x = conv_block(x, 512, convs=4, name="block4")
    x = MaxPool2D((2,2), strides=2, name='pool4')(x)

    x = conv_block(x, 512, convs=4, name="block5")
    x = MaxPool2D((2,2), strides=2, name='pool5')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=img, outputs=predictions, name='custom')
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def custom_x1(class_count):
    img = Input(shape=(224, 224, 3), name='input_layer')
    x = Lambda(img_channel_scale)(img)
    
    # 224 x 224

    x = conv_block(x, 32, convs=6, name="block1")
    x = MaxPool2D((4,4), strides=4, name='pool1')(x)

    # 56 x 56

    x = conv_block(x, 128, convs=8, name="block3")
    x = MaxPool2D((2,2), strides=2, name='pool3')(x)

    # 28 x 28

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=img, outputs=predictions, name='custom')
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def custom_x2(class_count):
    img = Input(shape=(224, 224, 3), name='input_layer')
    x = Lambda(img_channel_scale)(img)
    
    x = conv_block(x, 64, convs=2, name="block1")
    x = MaxPool2D((2,2), strides=2, name='pool1')(x)

    x = conv_block(x, 128, convs=2, name="block2")
    x = MaxPool2D((2,2), strides=2, name='pool2')(x)

    x = conv_block(x, 256, convs=4, name="block3")
    x = MaxPool2D((2,2), strides=2, name='pool3')(x)

    x = conv_block(x, 512, convs=4, name="block4")
    x = MaxPool2D((2,2), strides=2, name='pool4')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=img, outputs=predictions, name='custom')
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def custom_19a(class_count):
    img = Input(shape=(224, 224, 3), name='input_layer')

    x_img = Lambda(img_channel_scale)(img) 
    x = x_img

    attention_layer1 = DarkAttention()

    x = conv_block(x, 64, convs=2, name="block1")
    x = attention_layer1([x, img])
    x = MaxPool2D((2,2), strides=2, name='pool1')(x)

    attention_layer2 = DarkAttention()

    x = conv_block(x, 128, convs=2, name="block2")
    x = attention_layer2([x, img])
    x = MaxPool2D((2,2), strides=2, name='pool2')(x)

    attention_layer3 = DarkAttention()

    x = conv_block(x, 256, convs=4, name="block3")
    x = attention_layer3([x, img])
    x = MaxPool2D((2,2), strides=2, name='pool3')(x)

    attention_layer4 = DarkAttention()

    x = conv_block(x, 512, convs=4, name="block4")
    x = attention_layer4([x, img])
    x = MaxPool2D((2,2), strides=2, name='pool4')(x)

    attention_layer5 = DarkAttention()

    x = conv_block(x, 512, convs=4, name="block5")
    x = attention_layer5([x, img])
    x = MaxPool2D((2,2), strides=2, name='pool5')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=img, outputs=predictions, name='custom')
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def custom_mask1(class_count):
    img = Input(shape=(224, 224, 3), name='input_layer')

    x_img = Lambda(img_channel_scale)(img) 
    x = x_img

    mask_layer = MaskLayer()

    x = mask_layer(x)

    x = conv_block(x, 64, convs=2, name="block1")
    x = MaxPool2D((2,2), strides=2, name='pool1')(x)

    x = conv_block(x, 128, convs=2, name="block2")
    x = MaxPool2D((2,2), strides=2, name='pool2')(x)

    x = conv_block(x, 256, convs=4, name="block3")
    x = MaxPool2D((2,2), strides=2, name='pool3')(x)

    x = conv_block(x, 512, convs=4, name="block4")
    x = MaxPool2D((2,2), strides=2, name='pool4')(x)

    x = conv_block(x, 512, convs=4, name="block5")
    x = MaxPool2D((2,2), strides=2, name='pool5')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=img, outputs=predictions, name='custom')
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def custom_mask2(class_count):
    img = Input(shape=(224, 224, 3), name='input_layer')

    x_img = Lambda(img_channel_scale)(img) 
    x = x_img

    mask_layer = MaskLayer()

    x = mask_layer(x)

    x = conv_block(x, 32, convs=2, name="block1")
    x = MaxPool2D((2,2), strides=2, name='pool1')(x)

    x = conv_block(x, 64, convs=2, name="block2")
    x = MaxPool2D((2,2), strides=2, name='pool2')(x)

    x = conv_block(x, 128, convs=4, name="block3")
    x = MaxPool2D((2,2), strides=2, name='pool3')(x)

    x = conv_block(x, 256, convs=4, name="block4")
    x = MaxPool2D((2,2), strides=2, name='pool4')(x)

    x = conv_block(x, 256, convs=4, name="block5")
    x = MaxPool2D((2,2), strides=2, name='pool5')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=img, outputs=predictions, name='custom')
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


architectures = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'DenseNet121': DenseNet121,
    'MobileNetV3Small': MobileNetV3Small,
    'ResNet101V2': ResNet101V2,
    'custom_19': custom_19,
    'custom_19a': custom_19a,
    'custom_mask1': custom_mask1,
    'custom_mask2': custom_mask2,
    'custom_x1': custom_x1,
    'custom_x2': custom_x2
}

def build_model(name, class_count):
    if name not in architectures.keys():
        return None

    return architectures[name](class_count)

def load_model(name, class_count, dir_models):
    if name not in architectures.keys():
        print(f'Model {name} is not known.')
        return None
    
    model = architectures[name](class_count)
    model.summary()

    model.load_weights(f'{dir_models}{name}.weights.h5')

    return model
