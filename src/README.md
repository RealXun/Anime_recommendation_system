# Anime Recommendation systems based on Data from MyAnimeList
--------------------------------------
<p align="center">
    <img src="https://github.com/RealXun/EDA_ANIME/blob/main/src/images/Cover_images/cover.png" width="1000">
</p>

The goal of this project to create 3 types of Anime recommendation system. But, what is a Anime recommendation system? is a type of recommendation system that is specifically designed to suggest Anime titles to users based on their preferences. This system uses various algorithms and data analysis techniques to analyze user behavior, interests, and interactions with different Anime titles, and then recommends titles that the user may be interested in watching. The system typically works by analyzing a user's viewing history and rating history to determine their preferences. It may also consider other factors such as the user's demographic information, the popularity of the Anime title, and the similarity between different Anime titles. Once the system has analyzed this data, it generates a list of recommended Anime titles for the user to watch. These recommendations may be based on user ratings and viewing habits, as well as other factors such as the similarity between different Anime titles or the popularity of a particular title.


## Unsupervised Collaborative Filtering based on ratings Using k-Nearest Neighbors (kNN)
--------------------------------------
Collaborative Filtering is a technique used in recommendation systems, which aims to predict user preferences based on their historical behavior or preferences. In anime recommendation systems, Collaborative Filtering can be used to recommend new anime to users based on their ratings and preferences.

k-Nearest Neighbors (kNN) is a popular algorithm used in Collaborative Filtering. The kNN algorithm works by finding the k most similar users to a target user based on their ratings. Once the k most similar users have been identified, he algorithm recommends anime that have high ratings among those users.


## Unsupervised content based recommendation system
--------------------------------------
An unsupervised content-based recommendation system is a type of recommendation system that uses the features of items to recommend similar items to users. This approach is unsupervised because it doesn't require explicit feedback from users to make recommendations.

The basic idea behind a content-based recommendation system is to analyze the attributes or characteristics of items (such as movies, music, or books) and then recommend items that are similar to those that a user has already shown interest in. For example, if a user likes action movies, the recommendation system might recommend other action movies with similar characteristics, such as fast-paced plots and explosive special effects.

To create a content-based recommendation system, the first step is to gather data about the items being recommended. This data might include information such as the genre, actors, director, release date, and plot summary for movies or the artist, album, genre, and song lyrics for music.

Once the data has been collected, the next step is to analyze it to identify patterns and similarities between items. This can be done using machine learning techniques such as clustering, dimensionality reduction, or classification. The resulting model can then be used to recommend items to users based on their preferences.

One advantage of a content-based recommendation system is that it can work well even with sparse or incomplete user data, since it doesn't rely on user feedback to make recommendations. However, it may be less effective in situations where users have diverse interests or where there are not enough attributes to accurately capture the essence of the items being recommended.


## Supervised Collaborative Filtering based on ratings Using SVD method
--------------------------------------
Supervised Collaborative Filtering based on ratings is a recommendation system method that predicts user preferences based on the ratings of similar users, along with additional data sources. It is called supervised because it relies on a training dataset to learn the patterns of user behavior.

The user-item matrix in Collaborative Filtering represents the users' ratings for various items. In Supervised Collaborative Filtering based on ratings, the model is trained on the historical data of user-item interactions, along with additional data sources such as demographic information, search queries, or purchase history. The model learns the patterns of user behavior and uses this information to predict user preferences for items that they have not yet interacted with.

Singular Value Decomposition (SVD) is a matrix factorization technique used in Collaborative Filtering to reduce the dimensionality of the user-item matrix. SVD can decompose a large matrix into smaller matrices that capture the underlying relationships between users and items. In Supervised Collaborative Filtering based on ratings using SVD, the user-item matrix is decomposed into three smaller matrices: U, S, and V. U represents the users' preferences, S represents the singular values, and V represents the items' features.

The model is trained on a training dataset and evaluated on a testing dataset. The performance is evaluated using metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Precision, Recall, and F1-score. The SVD-based Supervised Collaborative Filtering method can be implemented using libraries such as Surprise in Python. As we do below


## Project Structure
--------------------------------------
```
src
  data
    _raw
    processed
    saved_models
      baseline
      test_models
  images
  notebooks
  scripts
    files
  utils
```

## URL to the recommendations system using Streamlit
--------------------------------------
https://realxun-anime-streamlit-srcapp-pqsjvd.streamlit.app/
  
  
  
