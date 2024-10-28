#!/usr/bin/env python
# coding: utf-8

# # Gender Identification based on Gait Analysis

# ## Exploratory Data Analysis 

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



male_g_fold = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Male_New'
female_g_fold = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Female_New'


def loading_folder_images(fold, lab):
    img_ar = []
    g_lab = []
    for f in os.listdir(fold):
        path_of_image = os.path.join(fold, f)
        images = cv2.imread(path_of_image)
        if images is not None:
            images = cv2.resize(images, (224, 224))  
            img_ar.append(images)
            g_lab.append(lab)
    return img_ar, g_lab


images_of_male_g, labels_of_male_g = loading_folder_images(male_g_fold, 0)
images_of_female_g, labels_of_female_g = loading_folder_images(female_g_fold, 1)


gender_images = np.array(images_of_male_g + images_of_female_g)
gender_labels = np.array(labels_of_male_g + labels_of_female_g)


# In[ ]:


def image_plots(gender_images, gender_labels, number_of_samples=6):
    plt.figure(figsize=(10, 5))
    for p in range(number_of_samples):
        plt.subplot(2, number_of_samples, p + 1)
        plt.imshow(cv2.cvtColor(gender_images[p], cv2.COLOR_BGR2RGB))
        plt.title('Male_image' if gender_labels[p] == 0 else 'Female_image')
        plt.axis('off')
        
        plt.subplot(2, number_of_samples, number_of_samples + p + 1)
        plt.imshow(cv2.cvtColor(gender_images[-(p + 1)], cv2.COLOR_BGR2RGB))
        plt.title('Male_image' if gender_labels[-(p + 1)] == 0 else 'Female_image')
        plt.axis('off')
    plt.show()

image_plots(gender_images, gender_labels)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_contain_data = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\subjects-info-pub.txt'


data_file = pd.read_csv(file_contain_data, sep='\t')


print(data_file.head())


# In[ ]:


plt.figure(figsize=(8, 5))
sns.countplot(data=data_file, x='Gender')
plt.title('Distribution of Gender')
plt.xlabel('Gender Types')
plt.ylabel('Number of Counts')
plt.show()


# # Deep Learning Model

# ## Convolutional Neural Network (CNN)

# In[ ]:


import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


image_size = (128, 128)
size_of_batch = 32
num_of_epochs = 2


# In[ ]:


def images_for_CNN(g_fold, gend_label):
    array_of_images = []
    gender_labels = []
    for n_f in os.listdir(g_fold):
        path_of_images = os.path.join(g_fold, n_f)
        gen_img = cv2.imread(path_of_images)  # Load image
        if gen_img is not None:
            gen_img = cv2.resize(gen_img, image_size)  # Resize for consistency
            gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            gen_img = img_to_array(gen_img)
            array_of_images.append(gen_img)
            gender_labels.append(gend_label)
    return array_of_images, gender_labels


# In[ ]:


cnn_m_images, cnn_m_labels = images_for_CNN(r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Male_New', 0)
cnn_f_images, cnn_f_labels = images_for_CNN(r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Female_New', 1)


comb_g_imgs = np.concatenate((cnn_m_images, cnn_f_images), axis=0)
comb_g_labs = np.concatenate((cnn_m_labels, cnn_f_labels), axis=0)


comb_g_imgs = comb_g_imgs / 255.0
comb_g_imgs = comb_g_imgs.reshape(-1, image_size[0], image_size[1], 1)  


from sklearn.preprocessing import LabelEncoder
comb_g_labs = to_categorical(comb_g_labs, num_classes=2)



X_train, X_test, y_train, y_test = train_test_split(comb_g_imgs, comb_g_labs, test_size=0.2, random_state=42)


# In[ ]:


data_augment = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
data_augment.fit(X_train)


# In[ ]:


cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

cnn_early_s = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


cnn_hist = cnn_model.fit(
    data_augment.flow(X_train, y_train, batch_size=size_of_batch),
    epochs=num_of_epochs,
    validation_data=(X_test, y_test),
    callbacks=[cnn_early_s]
)


# In[ ]:


cnn_validate = cnn_model.predict(X_test)
cnn_valid_predict_c = np.argmax(cnn_validate, axis=1)
cnn_valid_true_c = np.argmax(y_test, axis=1)


# In[ ]:


print("The CNN Classification Report:")
print(classification_report(cnn_valid_true_c, cnn_valid_predict_c, target_names=['Male Images', 'Female Images']))


# In[ ]:


cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {cnn_accuracy * 100:.2f}%")


# In[ ]:


cnn_train_pred = cnn_model.predict(X_train)
cnn_train_rmse = np.sqrt(mean_squared_error(y_train.argmax(axis=1), cnn_train_pred.argmax(axis=1)))


# In[ ]:


cnn_val_pred = cnn_model.predict(X_test)
cnn_valid_rmse = np.sqrt(mean_squared_error(y_test.argmax(axis=1), cnn_val_pred.argmax(axis=1)))


# In[ ]:


print(f"Train RMSE: {cnn_train_rmse:.4f}")
print(f"Validation RMSE: {cnn_valid_rmse:.4f}")


# In[ ]:


cnn_t_rmse_his = np.sqrt(cnn_hist.history['loss'])
cnn_v_rmse_his = np.sqrt(cnn_hist.history['val_loss'])


# In[ ]:


plt.plot(cnn_t_rmse_his, label='Train RMSE')
plt.plot(cnn_v_rmse_his, label='Validation RMSE')
plt.xlabel('num of Epochs')
plt.ylabel('value of RMSE')
plt.legend()
plt.title('The CNN RMSE Curves')
plt.show()


# ## Tuning Hyperparameters for CNN (Manually)

# In[ ]:


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical


image_size = (128, 128)


male_g_fold = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Male_New'
female_g_fold = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Female_New'


def images_for_CNN_hyp(g_fold, gend_label):
    array_of_images = []
    gender_labels = []
    for n_f in os.listdir(g_fold):
        path_of_images = os.path.join(g_fold, n_f)
        gen_img = cv2.imread(path_of_images)  # Load image
        if gen_img is not None:
            gen_img = cv2.resize(gen_img, image_size)  # Resize for consistency
            gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            gen_img = img_to_array(gen_img)
            array_of_images.append(gen_img)
            gender_labels.append(gend_label)
    return array_of_images, gender_labels




cnn_m_images, cnn_m_labels = images_for_CNN_hyp(male_g_fold, 0)
cnn_f_images, cnn_f_labels = images_for_CNN_hyp(female_g_fold, 1)


comb_g_imgs = np.concatenate((cnn_m_images, cnn_f_images), axis=0)
comb_g_labs = np.concatenate((cnn_m_labels, cnn_f_labels), axis=0)


comb_g_imgs = comb_g_imgs / 255.0


comb_g_labs = to_categorical(comb_g_labs, num_classes=2)



# In[ ]:


cnn_parameters = {
    'filters1': [32, 64],
    'filters2': [64, 128],
    'filters3': [128, 256],
    'kernel_size': [(3, 3), (5, 5)],
    'pool_size': [(2, 2), (3, 3)],
    'optimizer': ['adam', 'sgd'],
    'batch_size': [16, 32],
    'epochs': [10, 20]
}


results = {}


# In[ ]:


def cnn_model_hype(filters1=32, filters2=64, filters3=128, kernel_size=(3, 3), pool_size=(2, 2)):
    cnn_model = Sequential([
        Conv2D(filters1, kernel_size, activation='relu', input_shape=(image_size[0], image_size[1], 1)),
        MaxPooling2D(pool_size),
        Dropout(0.25),

        Conv2D(filters2, kernel_size, activation='relu'),
        MaxPooling2D(pool_size),
        Dropout(0.25),

        Conv2D(filters3, kernel_size, activation='relu'),
        MaxPooling2D(pool_size),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    cnn_early_s = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn_model


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


cnn_k_fold = KFold(n_splits=5, shuffle=True, random_state=42)


# In[ ]:


for i in cnn_parameters['filters1']:
    for j in cnn_parameters['filters2']:
        for k in cnn_parameters['filters3']:
            for ke in cnn_parameters['kernel_size']:
                for p in cnn_parameters['pool_size']:
                    cnn_high_accuracy = 0
                    for train_index, val_index in cnn_k_fold.split(images):
                        X_train, X_val = comb_g_imgs[train_index], comb_g_imgs[val_index]
                        y_train, y_val = comb_g_labs[train_index], comb_g_labs[val_index]

                        model = cnn_model_hype(i, j, k, kernel_size=ke, pool_size=p)
                        model.fit(data_augment.flow(X_train, y_train, batch_size=size_of_batch),epochs=num_of_epochs,validation_data=(X_test, y_test),callbacks=[cnn_early_s])

                        val_pred = model.predict(X_val)
                        val_pred_classes = np.argmax(val_pred, axis=1)
                        val_true_classes = np.argmax(y_val, axis=1)
                        cnn_accuracy = accuracy_score(val_true_classes, val_pred_classes)
                        if cnn_accuracy > cnn_high_accuracy:
                            cnn_high_accuracy = cnn_accuracy
                    cnn_configuration = f'filters1={i}, filters2={j}, filters3={k}, kernel_size={ke}, pool_size={p}'
                    results[cnn_configuration] = cnn_high_accuracy
                    print(f'{cnn_configuration} => Highest Accuracy of CNN: {cnn_high_accuracy:.4f}')



# # Machine Learning Models

# ## Random Forest

# In[ ]:


import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorly.decomposition import parafac
from tensorly import unfold, tensor
from sklearn.model_selection import GridSearchCV


# In[ ]:


radius_lbp = 3
num_LBP_pnts = 8 * LBP_RADIUS


size_of_image = (64,64)
NUM_COMPONENTS = 10


# In[ ]:


def images_loading_rf(fold, lab):
    img_ar = []
    g_lab = []
    for f in os.listdir(fold):
        path_of_image = os.path.join(fold, f)
        images = cv2.imread(path_of_image,cv2.IMREAD_GRAYSCALE)
        if images is not None:
            images = cv2.resize(images, size_of_image)  # Resize for consistency
            img_ar.append(images)
            g_lab.append(lab)
    return np.array(img_ar), np.array(g_lab)


# In[ ]:


rf_m_images, rf_m_labels = images_loading_rf(r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Male_New', 0)
rf_f_images, rf_f_labels = images_loading_rf(r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Female_New', 1)


comb_g_imgs = np.concatenate((rf_m_images, rf_f_images), axis=0)
comb_g_labs = np.concatenate((rf_m_labels, rf_f_labels), axis=0)


# In[ ]:


def rf_HOG_f(gend_imgs):
    rf_HOG_feat = []
    for i in gend_imgs:
        hog_feats = hog(i, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        rf_HOG_feat.append(hog_feats)
    return np.array(rf_HOG_feat)


# In[ ]:


def rf_LBP_f(gend_imgs):
    rf_LBP_feat = []
    for j in gend_imgs:
        loc_bi_p = local_binary_pattern(j, num_LBP_pnts, radius_lbp, method='uniform')
        histogrm, _ = np.histogram(loc_bi_p.ravel(), bins=np.arange(0, num_LBP_pnts + 3), range=(0, num_LBP_pnts + 2))
        histogrm = histogrm.astype("float")
        histogrm /= (histogrm.sum() + 1e-6)  # Normalize the histogram
        rf_LBP_feat.append(histogrm)
    return np.array(rf_LBP_feat)


# In[ ]:


def rf_PCA_f(gend_imgs, n_comp=50):
    rf_flat_imges = [k.flatten() for k in gend_imgs]
    rf_pca = PCA(n_components=n_comp)
    rf_pca_feat = rf_pca.fit_transform(rf_flat_imges)
    return rf_pca_feat


# In[ ]:


rf_HOG_features = rf_HOG_f(comb_g_imgs)
rf_LBP_features = rf_LBP_f(comb_g_imgs)
rf_PCA_features = rf_PCA_f(comb_g_imgs)


# In[ ]:


rf_comb_feats = np.concatenate((rf_HOG_features, rf_LBP_features, rf_PCA_features), axis=1)


X_train, X_test, y_train, y_test = train_test_split(rf_comb_feats, comb_g_labs, test_size=0.2, random_state=42)


# In[ ]:


rf_classif = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classif.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score

rf_y_predcts = rf_classif.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_predcts)
print(f"Random Forest Test Accuracy: {rf_accuracy * 100:.2f}%")
print(classification_report(y_test, rf_y_predcts))


# ## Tuning Hyperparameters for Random Forest using Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV

rf_parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


rand_forest_model = RandomForestClassifier(random_state=42)


rf_g_search = GridSearchCV(estimator=rand_forest_model, param_grid=rf_parameters, cv=3, n_jobs=-1, verbose=2)
rf_g_search.fit(X_train, y_train)


rf_best_parameters = rf_g_search.best_params_
print(f"Best parameters found: {rf_best_parameters}")


# In[ ]:


rand_f_model_opt = RandomForestClassifier(**rf_best_parameters, random_state=42)
rand_f_model_opt.fit(X_train, y_train)


rf_y_predctns = rand_f_model_opt.predict(X_test)


# In[ ]:


print("Classification Report of Random Forest:")
print(classification_report(y_test, rf_y_predctns, target_names=['Male Images', 'Female Images']))

rf_accuracy = accuracy_score(y_test, rf_y_predctns)
print(f"Test Accuracy of Random Forest: {rf_accuracy * 100:.2f}%")


# ## Decision Tree

# In[ ]:


import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.model_selection import GridSearchCV


# In[ ]:


def DT_Images_load(g_fold, gend_label, image_size = (224,224)):
    array_of_images = []
    gender_labels = []
    for n_f in os.listdir(g_fold):
        path_of_images = os.path.join(g_fold, n_f)
        gen_img = cv2.imread(path_of_images)  # Load image
        if gen_img is not None:
            gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)  
            gen_img = cv2.resize(gen_img, image_size)  # Resize for consistency
            gen_img = preprocess_input(gen_img)
            array_of_images.append(gen_img)
            gender_labels.append(gend_label)
    return np.array(array_of_images), np.array(gender_labels)


# In[ ]:


male_images_d = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Male_New'
female_images_d = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Female_New'
dt_m_images, dt_m_labels = DT_Images_load(male_images_d, 0)
dt_f_images, dt_f_labels = DT_Images_load(female_images_d, 1)


comb_g_imgs = np.concatenate((dt_m_images, dt_f_images), axis=0)
comb_g_labs = np.concatenate((dt_m_labels, dt_f_labels), axis=0)


X_train, X_test, y_train, y_test = train_test_split(comb_g_imgs, comb_g_labs, test_size=0.2, random_state=42)


# In[ ]:


dense_b_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')


X_trn_feat = dense_b_model.predict(X_train)
X_tst_feat = dense_b_model.predict(X_test)


# In[ ]:


dt_classif = DecisionTreeClassifier()
dt_classif.fit(X_trn_feat, y_train)


dt_y_predctns = dt_classif.predict(X_tst_feat)


dt_accuracy = accuracy_score(y_test, dt_y_predctns)
dt_cl_report = classification_report(y_test, dt_y_predctns, target_names=['Male images', 'Female images'])

print(f'Accuracy of the Model: {dt_accuracy}')
print(f'Classification Report of Decision Tree:\n{dt_cl_report}')


# ## Tuning Hyperparameters for Decision Tree using Grid Search

# In[ ]:


dt_parameters = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}


# In[ ]:


dt_g_search = GridSearchCV(estimator=dt_classif, param_grid=dt_parameters, cv=5, n_jobs=-1, verbose=2)


dt_g_search.fit(X_trn_feat, y_train)


dt_best_parameters = dt_g_search.best_params_
print(f'Best parameters found: {dt_best_parameters}')


# In[ ]:


dt_best_clf = dt_g_search.best_estimator_
dt_y_predctns = dt_best_clf.predict(X_tst_feat)


dt_accuracy = accuracy_score(y_test, dt_y_predctns)
dt_cl_report = classification_report(y_test, dt_y_predctns, target_names=['Male Images', 'Female Images'])


# In[ ]:


print(f'Accuracy: {dt_accuracy}')
print(f'Classification Report:\n{dt_cl_report}')


# ## K-nearest Neighbors (KNN)

# In[ ]:


import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.model_selection import GridSearchCV


# In[ ]:


def KNN_Images_load(g_fold, gend_label, image_size = (224,224)):
    array_of_images = []
    gender_labels = []
    for n_f in os.listdir(g_fold):
        path_of_images = os.path.join(g_fold, n_f)
        gen_img = cv2.imread(path_of_images)  # Load image
        if gen_img is not None:
            gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)  
            gen_img = cv2.resize(gen_img, image_size)  # Resize for consistency
            gen_img = preprocess_input(gen_img)
            array_of_images.append(gen_img)
            gender_labels.append(gend_label)
    return np.array(array_of_images), np.array(gender_labels)


# In[ ]:


male_d_images = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Male_New'
female_d_images = r'C:\Users\Rishwanth Mithra\Downloads\GEI Casia-B\Female_New'
knn_m_images, knn_m_labels = KNN_Images_load(male_d_images, 0)
knn_f_images, knn_f_labels = KNN_Images_load(female_d_images, 1)


comb_g_imgs = np.concatenate((male_images, female_images), axis=0)
comb_g_labs = np.concatenate((male_labels, female_labels), axis=0)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(comb_g_imgs, comb_g_labs, test_size=0.2, random_state=42)


resnt_b_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


rsnt_trn_feat = resnt_b_model.predict(X_train)
rsnt_tst_feat = resnt_b_model.predict(X_test)


# In[ ]:


knn_classif = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')


knn_classif.fit(rsnt_trn_feat, y_train)

knn_y_predctns = knn_classif.predict(rsnt_tst_feat)


knn_accuracy = accuracy_score(y_test, knn_y_predctns)
knn_cl_report = classification_report(y_test, knn_y_predctns, target_names=['Male Images', 'Female Images'])

print(f'KNN Accuracy: {knn_accuracy}')
print(f'Classification Report of KNN:\n{knn_cl_report}')


# ## Tuning Hyperparameters for KNN

# In[ ]:


Images_flat = comb_g_imgs.reshape(comb_g_imgs.shape[0], -1)  


X_train, X_test, y_train, y_test = train_test_split(Images_flat, comb_g_labs, test_size=0.2, random_state=42)


# In[ ]:


knn_parameters = {
    'n_neighbors': [3, 5, 7, 9], 
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


# In[ ]:


from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()
knn_g_search = GridSearchCV(knn_clf, knn_parameters, cv=5, scoring='accuracy')
knn_g_search.fit(X_train, y_train)


# In[ ]:


print("Best parameters found:", knn_g_search.best_params_)
print("Best KNN accuracy:", knn_g_search.best_score_)


# # Thank you
