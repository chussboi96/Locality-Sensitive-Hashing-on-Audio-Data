#!/usr/bin/env python
# coding: utf-8

# import glob
import pickle
import eyed3
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm


def write_file(name, data):
    with open(name, 'wb') as file:
        pickle.dump(data, file)


def read_file(name):
    with open(name, 'rb') as file:
        return pickle.load(file)


def feature_extraction(file):
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file, res_type='kaiser_fast')
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None


def unique_shingles(fvals):
    tot_shingles = list(fvals.values())

    shingles = []
    for i in tqdm(tot_shingles):  # tqdm shows the progress bar
        shingles.append(i)

    shingles = np.hstack(shingles)  # stacking the mfcc's horizontally
    shingles = np.array(list(dict.fromkeys(shingles)))  # .fromkeys will get the unique features

    return shingles


def finding_ones(fvals, shingles):  # checks if a shingle exits in the mfcc of current song
    return np.array([1 if x in fvals else 0 for x in shingles])  # 1 if exits else 0


def shingles_matrix(shingles, fvals):
    matrix = np.zeros(len(shingles))

    for v in tqdm(list(fvals.values())):
        matrix = np.vstack([matrix, finding_ones(v, shingles)])

    matrix = np.delete(matrix, (0), axis=0)  # matrix which has as rows the shingles and as columns the file names

    return matrix


def hash_matrix(matrix, shingles, names, perm):
    # shingles will be on the rows and file names as columns
    df = pd.DataFrame(matrix.transpose(), index=range(len(shingles)), columns=names)

    hash_matrix = np.zeros(len(names), dtype=int)

    # for permutation .sample is used to randomly shuffle the matrix
    # the first row index where 1 occurs will be stored in a list.
    # by stacking the list at each permutation we get the hash matrix

    for i in tqdm(range(perm)):
        hash_matrix = np.vstack(
            [hash_matrix, list(df.sample(frac=1, random_state=i).reset_index(drop=True).ne(0).idxmax())])

    hash_matrix = np.delete(hash_matrix, (0), axis=0)
    hash_mat = pd.DataFrame(hash_matrix, index=range(1, perm + 1), columns=names)
    return hash_mat


def create_buckets(hash_mat, div, perm):
    rows = int(perm / div)
    buckets = {}

    for name, hashval in hash_mat.iteritems():  # producing a label object and column object of dataframe
        hashval = list(hashval)  # convert each column of the dataframe into separate lists

        for i in range(0, len(hashval), rows):
            bucket_hash = tuple(hashval[
                                i: i + rows])  # select the first 5  rows and convert them into a tuple which will also be the bucketid

            if bucket_hash in buckets:
                buckets[bucket_hash].add(name)  # append file name as value if band is already present
            else:
                buckets[bucket_hash] = {name}  # otherwise make a new key and value

    return buckets


def Jaccard(A, B):
    # Find intersection of two sets
    nominator = A.intersection(B)

    # Find union of two sets
    denominator = A.union(B)

    # Take the ratio of sizes
    similarity = len(nominator) / len(denominator)

    return similarity


def query(file, buckets, hash_mat, shingles, perm, bands):
    score = (0, ' ')
    qfeatures = {}
    qfeatures[file] = feature_extraction(file)
    qshingles = unique_shingles(qfeatures)
    qmatrix = finding_ones(np.array(qshingles), shingles)

    qdf = pd.DataFrame(qmatrix.transpose(), index=range(len(shingles)), columns=['query'])
    hash_query = np.zeros(1, dtype=int)
    for i in range(perm):
        hash_query = np.vstack(
            [hash_query, list(qdf.sample(frac=1, random_state=i).reset_index(drop=True).ne(0).idxmax())])
    hash_query = np.delete(hash_query, (0), axis=0)
    hash_query = pd.DataFrame(hash_query, index=range(1, perm + 1), columns=['query'])

    rows = int(perm / bands)
    qbuckets = {}
    qhashval = list(hash_query['query'])
    for i in range(0, perm, rows):
        qbuckets[tuple(qhashval[i: i + rows])] = 'query'

    common = set(qbuckets).intersection(set(buckets))

    for band in common:
        for song in buckets[band]:
            jac = Jaccard(set(hash_query['query']), set(hash_mat[song]))
            if score <= (jac, song):
                score = (jac, song)  # store the maximum score

    return score


def get_details(score, features):
    if score[0]>0:
        file=score[1]
        for path in features.keys():
            if file in path:
                audio=eyed3.load(path)
                return audio.tag.title + " by " + audio.tag.artist, score[0]
            else:
                return "The provided does not match any songs. ", score[0]
