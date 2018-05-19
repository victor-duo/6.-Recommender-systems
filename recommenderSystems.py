from math import sqrt
from numpy import genfromtxt
import pandas as pd

def load_movie_data():
    data = pd.read_csv("movies.csv", quotechar='"', skipinitialspace=True)
    output = {}
    for index,row in data.iterrows():
        genres = {"Action":0,
                 "Adventure":0,
                 "Animation":0,
                 "Children":0,
                 "Comedy":0,
                 "Crime":0,
                 "Documentary":0,
                 "Drama":0,
                 "Fantasy":0,
                 "Film-Noir":0,
                 "Horror":0,
                 "Musical":0,
                 "Mystery":0,
                 "Romance":0,
                 "Sci-Fi":0,
                 "Thriller":0,
                 "War":0,
                 "Western":0}
        row[2] = row[2].split("|")
        for genre in row[2]:
            if(genre in genres):
                genres[genre] += 1
        output[row[0]] = [row[1],genres]
    # for row in output:
    #     print(row)
    return output

def load_user_data(movieData):
    data = pd.read_csv("ratings.csv")
    output = []
    userMovie = {} # dict of users with array of movies as values
    userId = 1
    movies = []
    for index,row in data.iterrows():
        if(userId != row[0] or len(data)-1 == index):
            userMovie[userId] = movies
            movies = []
            userId = row[0]
        if(row[2] >= 2.5):
            movies.append(row[1])
    for user in userMovie:
        genres = {"Action":0,
                 "Adventure":0,
                 "Animation":0,
                 "Children":0,
                 "Comedy":0,
                 "Crime":0,
                 "Documentary":0,
                 "Drama":0,
                 "Fantasy":0,
                 "Film-Noir":0,
                 "Horror":0,
                 "Musical":0,
                 "Mystery":0,
                 "Romance":0,
                 "Sci-Fi":0,
                 "Thriller":0,
                 "War":0,
                 "Western":0}
        for movie in userMovie[user]: # movies rated by the user
            for genre in movieData[movie][1]:
                if(movieData[movie][1][genre] != 0):
                    genres[genre] += 1
        output.append([user,genres])
    return output, userMovie

movieData = load_movie_data()
# movieData is a dict by movieId as key and as value we have an array
# composed of movieName and dict of genres
usersVect, movieIDRatedByUsers = load_user_data(movieData)
# movieIDRatedByUsers is necessary for recommendations of item that the user
# has not rated yet
# now that we used movieData we can convert it to a movieVector
movieVect = []
for key in movieData:
    movieInfo = movieData[key]
    movieVect.append([movieInfo[0],movieInfo[1]])

def load_data_for_CF():
    data = genfromtxt('ratings.csv', delimiter=',', dtype=None)
    data = data[1:len(data)] # remove the header
    output = []
    itemRatings = {}
    curUser = data[0][0]
    for tab in data:
        if(curUser != tab[0]):
            output.append([curUser,itemRatings])
            itemRatings = {}
            curUser = tab[0]
        itemRatings[int(tab[1])] = float(tab[2])
    output.append([curUser,itemRatings])
    return output


dataCF = load_data_for_CF()

def user_sim_cosine_sim(person1, person2):
# computes similarity between two users based on the cosine similarity metric
    res = 0
    normA = 0
    normB = 0
    for genre in person1[1]:
        res += person1[1][genre]*person2[1][genre]
        normA += person1[1][genre]*person1[1][genre]
        normB += person2[1][genre]*person2[1][genre]
    normA = sqrt(normA)
    normB = sqrt(normB)
    if(normA*normB == 0): return 0
    res = res/(normA*normB)
    return res

def user_sim_cosine_sim_CF(person1, person2):
    res = 0
    normA = 0
    normB = 0
    for movie in person1[1]:
        if(movie in person2[1]):
            res += person1[1][movie]*person2[1][movie]
    for movie in person1[1]:
        normA += person1[1][movie]*person1[1][movie]
    for movie in person2[1]:
        normB += person2[1][movie]*person2[1][movie]
    normA = sqrt(normA)
    normB = sqrt(normB)
    if(normA*normB == 0): return 0
    res = res/(normA*normB)
    return res

def user_sim_pearson_corr_CF(person1, person2):
# computes similarity between two users based on the pearson similarity metric
    mean1 = 0
    mean2 = 0
    nbOfValues1 = len(person1[1])
    nbOfValues2 = len(person2[1])
    res = 0
    for movie1 in person1[1]:
        mean1 += person1[1][movie1]
    for movie2 in person2[1]:
        mean2 += person2[1][movie2]
    mean1 /= float(nbOfValues1)
    mean2 /= float(nbOfValues2)
    A = 0
    B = 0
    norm1 = 0
    norm2 = 0
    for movie in person1[1]:
        if(movie in person2[1]):
            A += (person1[1][movie]-mean1)*(person2[1][movie]-mean2)
            norm1 += (person1[1][movie]-mean1)*(person1[1][movie]-mean1)
            norm2 += (person2[1][movie]-mean2)*(person2[1][movie]-mean2)
    norm1 = sqrt(norm1)
    norm2 = sqrt(norm2)
    B = norm1*norm2
    if(B == 0): return 0
    res = A/B
    return res

def most_similar_users(person, number_of_users):
# returns top-K similar users for the given user
# K = 2
    cosSimTab = []
    for i in range(number_of_users):
        if(person[0]!=dataCF[i][0]):
            cosSimTab.append([i,
                             user_sim_cosine_sim_CF(person,dataCF[i])])
    cosSimTab = sorted(cosSimTab, key=lambda value:value[1], reverse=True)[:2]
    print(cosSimTab)
    top2Users = [dataCF[cosSimTab[0][0]], dataCF[cosSimTab[1][0]]]
    # print(top2Users)
    return top2Users

def user_recommendations_content_based(person):
    itemSimilarity = []
    movieIDRatedByPerson = movieIDRatedByUsers[person[0]]
    for id in movieData:
        movieName = movieData[id][0]
        if(id not in movieIDRatedByPerson):
            for i in range(len(movieVect)):
                if(movieVect[i][0] == movieName):
                    correspondingMovieVect = movieVect[i]
            cosineSimilarity = user_sim_cosine_sim(correspondingMovieVect, person)
            itemSimilarity.append([correspondingMovieVect[0],cosineSimilarity])
    itemSimilarity = sorted(itemSimilarity, key = lambda value:value[1],
                            reverse=True)[:3]
    return itemSimilarity

resultFromContentBased = user_recommendations_content_based(usersVect[3])
print(resultFromContentBased)

def user_recommendationsCF(person):
# generate recommendations for the given user
    top2Users = most_similar_users(person,len(dataCF))
    item = []
    for id in movieData:
        if(id not in person[1] and
           id in top2Users[0][1] and
           id in top2Users[1][1]):
           # unrated item
           pearson1 = user_sim_pearson_corr_CF(top2Users[0], person)
           pearson2 = user_sim_pearson_corr_CF(top2Users[1], person)
           if(pearson1+pearson2 != 0):
               predRating = (top2Users[0][1][id]*pearson1+top2Users[1][1][id]*
                             pearson2)/(pearson1+pearson2)
           else: predRating = 0
           item.append([id,predRating])
    item = sorted(item, key=lambda value:value[1], reverse=True)[:3]
    return item

resultFromCF = user_recommendationsCF(dataCF[3])
print(resultFromCF)
for result in resultFromCF:
    result[1] = result[1]/5
print(resultFromCF)
# to see the name from the CF result
for result in resultFromCF:
    print("name: {}".format(movieData[result[0]][0]))
