# To install dependencies, run (in this folder):
#   pip install -r requirements.txt
#%%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("razanaqvi14/real-and-fake-news")

print("Path to dataset files:", path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # plottinng in python library
from sklearn.preprocessing import LabelEncoder # for label encoding
from sklearn.model_selection import train_test_split # to split our dataset in training and test set
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from itertools import islice
from tensorflow.keras.utils import pad_sequences
from getpass import getpass

username = getpass("Enter username: ")

df_true = pd.read_csv(f"/Users/{username}/.cache/kagglehub/datasets/razanaqvi14/real-and-fake-news/versions/1/True.csv")
df_fake = pd.read_csv(f"/Users/{username}/.cache/kagglehub/datasets/razanaqvi14/real-and-fake-news/versions/1/Fake.csv")

# label data
df_true['label'] = 'true'
df_fake['label'] = 'fake'

# Concattenating data
combined = pd.concat([df_true, df_fake], ignore_index=True)

# Exploring dataset
# print(combined.head())
# print(combined.shape)
# print(combined.describe())
print(combined.columns.to_list)
print(combined['label'].sample(10))

# Bar chart by classes
label_counts = combined['label'].value_counts()
count_true = label_counts.get('true', 0)
count_false = label_counts.get('fake', 0)

# plt.bar(['True', 'Fake'], [count_true, count_false])
# plt.xlabel('Label')
# plt.ylabel('Quantity')
# plt.title('Quantity of True and Fake News')
# plt.show()

fake_prob = np.round((count_false / (count_false + count_true))*100, 2)
print("Fake news probability: ", fake_prob)

# Label Encoder
label_encoder = LabelEncoder()
combined['label'] = label_encoder.fit_transform(combined['label'])
print(combined.head(5))

# Splitting to y and X
X = combined['text'].to_list()
y = combined['label'].to_list()

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

print(take(10, tokenizer.word_index.items()))

X = pad_sequences(X, maxlen=745)

X = np.array(X)
y = np.array(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

values, counts = np.unique(y_test, return_counts=True)
for value, count in zip(values, counts):
    print(f'Value: {value} - Count: {count}')

print(f'X_train_shape: {X_train.shape}')

vocab_size = len(tokenizer.word_index) + 1
print(f'The vocabulary consists of {vocab_size}')

# Improved model: Embedding + LSTM + Dense layers with Dropout
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping  # Early Stopping vorerst deaktiviert
import os

model_path = "fake_news_model_v3.keras"

# Wenn ein gespeichertes Modell existiert, lade es
if os.path.exists(model_path):
    print(f"Lade vorhandenes Modell aus {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    # Kurzes Nachtrainieren, damit immer eine History vorhanden ist
    history = model.fit(
        X_train,
        y_train,
        epochs=1,              # nur 1 Epoche
        batch_size=128,
        validation_split=0.2,
        # callbacks=[early_stopping]
    )

else:
    print("Kein gespeichertes Modell gefunden – Modell wird trainiert.")

    model = tf.keras.Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=745),
        SpatialDropout1D(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    model.summary()

    # Early Stopping ist auskommentiert
    # early_stopping = EarlyStopping(
    #     monitor='val_loss',
    #     patience=2,
    #     restore_best_weights=True
    # )

    history = model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=128,
        validation_split=0.2,
        # callbacks=[early_stopping]
    )

    # Modell nach dem Training speichern
    model.save(model_path)
    print(f"Modell wurde in {model_path} gespeichert.")

# Ab hier gibt es IMMER eine history, also immer plotten
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('Test evaluation:')
model.evaluate(X_test, y_test)

new_text_real = ['U.S. Transportation Secretary Sean Duffy issued a stern warning to New York Governor Kathy Hochul on Monday, stating that the state “risks serious consequences” if it continues to enforce its congestion pricing initiative in Manhattan. The congestion pricing program, which was approved by the Biden administration and launched on January 5, charges a $9 toll on most passenger vehicles entering Manhattan south of 60th Street during peak hours. Designed to reduce traffic, improve air quality, and generate revenue for public transit, the system mirrors similar programs in cities like London and is popular among environmental advocates. However, in a letter dated Monday, the Trump administration reiterated its opposition to the program, demanding that Governor Hochul halt toll collection immediately. The letter gave her until May 21 to certify that tolls have ceased or to explain why continuing the program does not violate federal law. “I write to warn you that the State of New York risks serious consequences if it continues to fail to comply with Federal law,” Duffy wrote. “President Trump and I will not sit back while Governor Hochul engages in class warfare and prices working-class Americans out of accessing New York City.” Duffy emphasized that the federal government provides billions in funding to New York and warned that those funds could be in jeopardy. “We are giving New York one last chance to turn back or prove their actions are not illegal,” he said. The administration threatened that if tolling continues past May 28, the federal government may begin withholding transportation funds and block approvals for future infrastructure projects in the state. This is not the first ultimatum from the Trump administration. Similar deadlines were issued in March and April, but the state refused to comply. Governor Hochul and the Metropolitan Transit Authority (MTA), which operates the program, have defended the initiative, claiming it is already showing positive results. In March, Hochul reported that traffic had decreased by 11% compared to the previous year, and travel times across bridges and tunnels had improved by 30%. Commuters entering the zone are reportedly saving up to 21 minutes per trip. “The program is doing what we hoped: traffic is down and business is up,” Hochul said. MTA Chair and CEO Janno Lieber added, “Congestion relief is working—cars and buses are moving faster, foot traffic is up, and even noise complaints are down.” Financially, the program has also exceeded expectations. In its first month, it raised $48.6 million in tolls, according to the New York Times. The MTA projects it will generate $500 million by year’s end. March data showed about 560,000 vehicles entered the congestion zone daily, a 13% reduction from the estimated 640,000 that would have entered without tolls. Public opinion remains mixed. A March NBC New York survey found that 42% of New York City residents support the tolls, while 35% back Trump’s efforts to dismantle them. Statewide, support drops to around 33%, with 40% favoring the program’s termination. In February, the Trump administration announced it was revoking federal approval for the program. However, the MTA has challenged the decision in federal court. In a recent ruling, a judge dismissed several claims filed by opponents, including the local trucking industry, further bolstering the program’s legal standing. As the May deadlines loom, tensions between state and federal authorities continue to escalate, putting the future of New York’s congestion pricing plan—and its transportation funding—at stake.', 'In what could be the most astonishing discovery of the century, a group of independent researchers claims to have uncovered the remains of an ancient, highly advanced city buried deep beneath the Antarctic ice. According to leaked drone footage and thermal scans obtained by the controversial group Polar Watch International, strange symmetrical structures, tunnels, and what appear to be energy-emitting obelisks lie under nearly two miles of solid ice near the Shackleton Range. The images, though grainy and unverified, have already sparked a storm of speculation online. Reddit forums, Twitter threads, and alternative news sites are flooded with theories ranging from a lost Atlantean outpost to remnants of extraterrestrial colonization. One particularly viral post claims the underground complex "pulsates with energy" and emits radio frequencies untraceable to any known source. Despite growing public curiosity, major media outlets have remained silent — leading some to accuse global powers of orchestrating a massive cover-up. The United Nations and several world governments have allegedly imposed strict no-fly zones over the region. In a statement released this morning, a spokesperson for the U.S. Department of Defense denied the existence of any such structures, calling the reports “baseless internet fabrications.” However, skeptics are not convinced. “Why are military planes flying over an empty patch of ice every day?” asked Dr. Linda Merrow, a geologist and whistleblower formerly affiliated with the U.S. Geological Survey. “The public deserves transparency.” Further intrigue surrounds a sudden expedition to Antarctica organized last week by a private aerospace firm with ties to several high-profile tech billionaires. Eyewitnesses report seeing cargo planes loaded with scientific equipment, security personnel, and unidentified black containers departing from a Chilean airbase. The firm has refused to comment. “This could change everything we know about human history,” said retired archaeologist Peter McAlister, who believes the structures resemble pre-Egyptian megalithic architecture. “If this city is real, it predates every known civilization.” Adding fuel to the fire, a group of tourists aboard a Russian research vessel claimed to have seen strange blue lights rising from the ice shelf late at night. Though dismissed by scientists as “atmospheric anomalies,” their video has garnered millions of views and raised further questions. Meanwhile, conspiracy theorists insist this is all part of a long-standing suppression effort known as the “Antarctica Protocol” — a secret agreement among world powers to prevent access to sensitive sites hidden under the polar ice. As pressure mounts, scientists, journalists, and internet sleuths continue to dig deeper, demanding access and accountability. Whether a hoax, a misunderstanding, or the discovery of an ancient alien metropolis, one thing is clear: the frozen continent may be hiding secrets far beyond what we imagined. For now, the world watches and waits — while Antarctica keeps her silence.']

new_text_real = tokenizer.texts_to_sequences(new_text_real)  # Tokenize the new text
new_text_real = pad_sequences(new_text_real, maxlen=745)  # Pad the tokenized sequences

prediction_real = model.predict(new_text_real)
print("Chances being real: ", prediction_real)

new_text_fake = ["Scientists reveal shocking discovery: drinking water triggers hidden mind control. An internal report claims trace minerals increase obedience and impulsive buying. Authorities deny everything, while supermarkets profit. Online witnesses share alarming stories, urging citizens to stop consumption immediately worldwide now."]

new_text_fake = tokenizer.texts_to_sequences(new_text_fake)  # Tokenize the new text
new_text_fake = pad_sequences(new_text_fake, maxlen=745)  # Pad the tokenized sequences

prediction_fake = model.predict(new_text_fake)
print("Chances being real: ", prediction_fake)
