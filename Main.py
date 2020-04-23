#!/usr/bin/env python
# coding: utf-8

# In[54]:


# Import Libary
import pandas as pd
import math
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import matplotlib.pyplot as plt

# read data
reviews = pd.read_csv("Hotel_Reviews.csv")

# append the positive and negative text reviews
reviews["review"] = reviews["reviews.title"] + "." + reviews["reviews.text"]

reviews["name"] = reviews["name"]

# create the label and make information easy to understanding
reviews["is_good_review"] = reviews["reviews.rating"].apply(lambda x: 1 if x > 3 else 0)

# If it bad is 0
# If it goof is 1

reviews = reviews[["review", "is_good_review", "name"]]
reviews.head()


# In[2]:


reviews.head(10)


# In[3]:


# return the wordnet object value corresponding to the POS tag
def get_wordnet_pos(pos_tag):
    
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    
    else:
        return wordnet.NOUN
    


# In[4]:


def clean_text(text):
    
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    
    # pos tag text
    pos_tags = pos_tag(text)
    
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    
    # join all
    text = " ".join(text)
    return(text)

# clean text data
#4768 
for i in range(len(reviews["review"])): reviews["review_clean"] = clean_text(reviews["review"][i])


# In[5]:


#add sentiment anaylsis columns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
reviews["sentiments"] = reviews["review"].apply(lambda x: sid.polarity_scores(x))
reviews = pd.concat([reviews.drop(['sentiments'], axis=1), reviews['sentiments'].apply(pd.Series)], axis=1)


# In[6]:


# add number of characters column
reviews["nb_chars"] = reviews["review"].apply(lambda x: len(x))

# add number of words column
reviews["nb_words"] = reviews["review"].apply(lambda x: len(x.split(" ")))


# In[7]:


#create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews["review_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec = reviews["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec.columns = ["doc2vec_vector_" + str(x) for x in doc2vec.columns]
reviews = pd.concat([reviews, doc2vec], axis=1)


# In[8]:


# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews.index
reviews = pd.concat([reviews, tfidf_df], axis=1)


# In[9]:


reviews["is_good_review"].value_counts(normalize = True)


# In[10]:


# wordcloud function

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(reviews["review"])


# In[11]:


# highest positive sentiment reviews (with more than 5 words)
reviews[reviews["nb_words"] >= 5].sort_values("pos", ascending = False)[["review", "pos"]].head(10)


# In[12]:


# lowest negative sentiment reviews (with more than 5 words)
reviews[reviews["nb_words"] >= 5].sort_values("neg", ascending = False)[["review", "neg"]].head(10)


# In[13]:


# plot sentiment distribution for positive and negative reviews

import seaborn as sns

for x in [0, 1]:
    subset = reviews[reviews['is_good_review'] == x]
    
    # Draw the density plot
    if x == 0:
        label = "Good reviews"
    else:
        label = "Bad reviews"
    sns.distplot(subset['compound'], hist = False, label = label)


# In[14]:


print(reviews)


# In[45]:


# data set category
cat_good = {
    "staff" : ["friendly", "nice", "good","helpful", "great", "polite", "professional", "excellent", "lovely", "courteous", 
               "courteous", "gentle", "gracious", "modest", "bland", "serviceable", "pleasant", "efficient", "engaging", 
               "wonderful", "attentive"],
    "room" : ["clean", "perfectly", "comfortable", "great", "comfort", "beautiful", "quiet", "well", "perfect", "extraordinary",
            "elegant", "privacy","private","safe", "safety", "big" ,"large", "spacious", "best", "convenient"],
    "location" : ["amazing", "great", "nice", "wonderful", "perfect", "perfectly", "best", "convenient"],
    "price" : ["cheap", "inexpensive", "good", "worth", "worthy"],
    "parking" : ["free", "near", "nice", "big"]
    }

# add colunm about category in 'reviews'
reviews["staff"] = 0
reviews["room"] = 0
reviews["location"] = 0
reviews["price"] = 0
reviews["parking"] = 0

# check category in 'reviews'
for i in range(len(reviews['review'])):
    
    lst = reviews["review"][i].lower().split(".")
    
    for text in lst:
        
    # check category about staff in reviews
        if "staff" in text:
            for cate in cat_good["staff"]:
                if cate in text:
                    reviews["staff"][i] += 1
    
    # check category about room in reviews
        if "room" in text:
            for cate in cat_good["room"]:
                if cate in text:
                    reviews["room"][i] += 1
                    
    # check category about location in reviews
        if "location" in text:
            for cate in cat_good["location"]:
                if cate in text:
                    reviews["location"][i] += 1
                    
    # check category about price in reviews
        if "price" in text:
            for cate in cat_good["price"]:
                if cate in text:
                    reviews["price"][i] += 1
    
    # check category about parking in reviews
        if "parking" in text:
            for cate in cat_good["parking"]:
                if cate in text:
                    reviews["parking"][i] += 1

reviews.head(15)


# In[49]:


# new dictionary have hotel's name (key) / category (values)
hotel_reviews = {}
all_hotel = []
count = 0

for i in range(len(reviews['review'])):
    
    # add name hotel
    if reviews["name"][i] not in all_hotel:
        hotel_reviews[count] = {"name" : reviews["name"][i], "staff" : 0, "room" : 0, "location" : 0, "price" : 0, "parking" : 0, "score" : 0}
        all_hotel.append(reviews["name"][i])
        count += 1
        
    # add values category hotel
    else:
        ind = all_hotel.index(reviews["name"][i]) # find index for find hotel

        hotel_reviews[ind]["staff"] += reviews["staff"][i]
        hotel_reviews[ind]["room"] += reviews["room"][i]
        hotel_reviews[ind]["location"] += reviews["location"][i]
        hotel_reviews[ind]["price"] += reviews["price"][i]
        hotel_reviews[ind]["parking"] += reviews["parking"][i]
        hotel_reviews[ind]["score"] += reviews["staff"][i] + reviews["parking"][i]                                                    + reviews["room"][i] + reviews["location"][i] + reviews["price"][i]
        
# sorted hotel reviews by score (high to low)
hotel_reviews = dict(sorted(hotel_reviews.items(), key=lambda kv: kv[1]['score'], reverse=True))

# เผื่อเอาไปใช้ ถ้าไม่ใช้ก็ลบได้ (ข้างล่างนี้)
# sorted hotel reviews by staff (high to low)
staff_reviews = dict(sorted(hotel_reviews.items(), key=lambda kv: kv[1]['staff'], reverse=True))

# sorted hotel reviews by room (high to low)
room_reviews = dict(sorted(hotel_reviews.items(), key=lambda kv: kv[1]['room'], reverse=True))

# sorted hotel reviews by location (high to low)
location_reviews = dict(sorted(hotel_reviews.items(), key=lambda kv: kv[1]['location'], reverse=True))

# sorted hotel reviews by price (high to low)
price_reviews = dict(sorted(hotel_reviews.items(), key=lambda kv: kv[1]['price'], reverse=True))

# sorted hotel reviews by parking (high to low)
parking_reviews = dict(sorted(hotel_reviews.items(), key=lambda kv: kv[1]['parking'], reverse=True))


# In[50]:


# Dictionary to Dataframe
hotel_reviews = pd.DataFrame.from_dict(hotel_reviews, orient='index')

#        staff room location price parking score
# name0    1    2      3       4     5       6
# name1    1    2      3       4     5       6
# name2    1    2      3       4     5       6
# name3    1    2      3       4     5       6

hotel_reviews.head()

# เผื่อเอาไปใช้ ถ้าไม่ใช้ก็ลบได้ (ข้างล่างนี้)
# sorted hotel reviews by staff (high to low)
staff_reviews = pd.DataFrame.from_dict(staff_reviews, orient='index')

# sorted hotel reviews by room (high to low)
room_reviews = pd.DataFrame.from_dict(room_reviews, orient='index')

# sorted hotel reviews by location (high to low)
location_reviews = pd.DataFrame.from_dict(location_reviews, orient='index')

# sorted hotel reviews by price (high to low)
price_reviews = pd.DataFrame.from_dict(price_reviews, orient='index')

# sorted hotel reviews by parking (high to low)
parking_reviews = pd.DataFrame.from_dict(parking_reviews, orient='index')


# In[51]:


# Top 10 Ranking Score to create bar graph (Sum All Score)

# List for name of Hotels
lst = [i for i in hotel_reviews['name']]
names = []
for i in range(10):
    names.append(lst[i])

# List for score of Staff
lst = [i for i in hotel_reviews['staff']]
staff = []
for i in range(10):
    staff.append(lst[i])

# List for score of Room
lst = [i for i in hotel_reviews['room']]
room = []
for i in range(10):
    room.append(lst[i])
    
# List for score of Location
lst = [i for i in hotel_reviews['location']]
location = []
for i in range(10):
    location.append(lst[i])
    
# List for score of Price
lst = [i for i in hotel_reviews['price']]
price = []
for i in range(10):
    price.append(lst[i])
    
# List for score of Parking
lst = [i for i in hotel_reviews['parking']]
parking = []
for i in range(10):
    parking.append(lst[i])


# In[55]:


# Create Stacked Bar Graph =w=
staff = np.array(staff)
room = np.array(room)
location = np.array(location)
price = np.array(price)
parking = np.array(parking)

# Fig size
plt.figure(figsize=(15, 7))

# Stacked Graph
plt.bar(names, staff, width=0.7, label='staff', color='#FFFF33', bottom=room+location+price+parking) # Staff
plt.bar(names, room, width=0.7, label='room', color='#FFCC33', bottom=location+price+parking)        # Room
plt.bar(names, location, width=0.7, label='location', color='#FF9933', bottom=price+parking)         # Location
plt.bar(names, price, width=0.7, label='price', color='#FF6633', bottom=parking)                     # Price
plt.bar(names, parking, width=0.7, label='parking', color='#FF0033')                                 # Parking

# Plot Graph
plt.suptitle("Ranking Hotel")
plt.xticks(rotation=90)
plt.xlabel("Hotels")
plt.ylabel("Score")
plt.legend(loc="upper right")

# Show Graph
plt.show()


# In[66]:


# เผื่อเอาไปใช้ ถ้าไม่ใช้ก็ลบได้ (ข้างล่างนี้)
# Top 10 Ranking Score to create bar graph (***Staff*** All Score)

# List for name of Hotels
lst = [i for i in staff_reviews['name']]
names = []
for i in range(10):
    names.append(lst[i])

# List for score of Staff
lst = [i for i in staff_reviews['staff']]
staff = []
for i in range(10):
    staff.append(lst[i])

# Create Stacked Bar Graph =w=
staff = np.array(staff)

# Fig size
plt.figure(figsize=(15, 7))

# Stacked Graph
plt.bar(names, staff, width=0.7, label='staff', color='#FFFF33') # Staff

# Plot Graph
plt.suptitle("Ranking Hotel")
plt.xticks(rotation=90)
plt.xlabel("Hotels")
plt.ylabel("Score")
plt.legend(loc="upper right")

# Show Graph
plt.show()


# In[67]:


# เผื่อเอาไปใช้ ถ้าไม่ใช้ก็ลบได้ (ข้างล่างนี้)
# Top 10 Ranking Score to create bar graph (***Room*** All Score)

# List for name of Hotels
lst = [i for i in room_reviews['name']]
names = []
for i in range(10):
    names.append(lst[i])

# List for score of Room
lst = [i for i in room_reviews['room']]
room = []
for i in range(10):
    room.append(lst[i])

# Create Stacked Bar Graph =w=
room = np.array(room)

# Fig size
plt.figure(figsize=(15, 7))

# Stacked Graph
plt.bar(names, room, width=0.7, label='room', color='#FFCC33') # Room

# Plot Graph
plt.suptitle("Ranking Hotel")
plt.xticks(rotation=90)
plt.xlabel("Hotels")
plt.ylabel("Score")
plt.legend(loc="upper right")

# Show Graph
plt.show()


# In[68]:


# เผื่อเอาไปใช้ ถ้าไม่ใช้ก็ลบได้ (ข้างล่างนี้)
# Top 10 Ranking Score to create bar graph (***Location*** All Score)

# List for name of Hotels
lst = [i for i in location_reviews['name']]
names = []
for i in range(10):
    names.append(lst[i])
    
# List for score of Location
lst = [i for i in location_reviews['location']]
location = []
for i in range(10):
    location.append(lst[i])

# Create Stacked Bar Graph =w=
location = np.array(location)

# Fig size
plt.figure(figsize=(15, 7))

# Stacked Graph
plt.bar(names, location, width=0.7, label='location', color='#FF9933') # Location

# Plot Graph
plt.suptitle("Ranking Hotel")
plt.xticks(rotation=90)
plt.xlabel("Hotels")
plt.ylabel("Score")
plt.legend(loc="upper right")

# Show Graph
plt.show()


# In[70]:


# เผื่อเอาไปใช้ ถ้าไม่ใช้ก็ลบได้ (ข้างล่างนี้)
# Top 10 Ranking Score to create bar graph (***Price*** All Score)

# List for name of Hotels
lst = [i for i in price_reviews['name']]
names = []
for i in range(10):
    names.append(lst[i])
    
# List for score of Price
lst = [i for i in price_reviews['price']]
price = []
for i in range(10):
    price.append(lst[i])

# Create Stacked Bar Graph =w=
price = np.array(price)

# Fig size
plt.figure(figsize=(15, 7))

# Stacked Graph
plt.bar(names, price, width=0.7, label='price', color='#FF6633') # Price

# Plot Graph
plt.suptitle("Ranking Hotel")
plt.xticks(rotation=90)
plt.xlabel("Hotels")
plt.ylabel("Score")
plt.legend(loc="upper right")

# Show Graph
plt.show()


# In[71]:


# เผื่อเอาไปใช้ ถ้าไม่ใช้ก็ลบได้ (ข้างล่างนี้)
# Top 10 Ranking Score to create bar graph (***Parking*** All Score)

# List for name of Hotels
lst = [i for i in parking_reviews['name']]
names = []
for i in range(10):
    names.append(lst[i])
    
# List for score of Parking
lst = [i for i in parking_reviews['parking']]
parking = []
for i in range(10):
    parking.append(lst[i])

# Create Stacked Bar Graph =w=
parking = np.array(parking)

# Fig size
plt.figure(figsize=(15, 7))

# Stacked Graph
plt.bar(names, parking, width=0.7, label='parking', color='#FF0033') # Parking

# Plot Graph
plt.suptitle("Ranking Hotel")
plt.xticks(rotation=90)
plt.xlabel("Hotels")
plt.ylabel("Score")
plt.legend(loc="upper right")

# Show Graph
plt.show()


# In[ ]:




