import pandas as pd


# read data

reviews = pd.read_csv("Hotel_Reviews.csv")

# append the positive and negative text reviews
reviews["review"] = reviews["Negative_Review"] + reviews["Positive_Review"]

# create the label and make information easy to understanding
reviews["is_bad_review"] = reviews["Reviewer_Score"].apply(lambda x: 1 if x < 5 else 0)

reviews = reviews[["review", "is_bad_review"]]
reviews.head()