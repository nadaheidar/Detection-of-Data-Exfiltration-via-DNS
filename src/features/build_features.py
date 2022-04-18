#imports
import tldextract
import pandas as pd
import csv
import math
from collections import Counter
import numpy as np

from src.Kafka.Consumer_kafka import consumer_data
#read domains data as dataframe
def build_features(df):
    #function ot count domain's charachters and generate new feature for it
    def count_FQDN(domain):
        count = len(domain)
        return count
    df['FQDN_count'] = [count_FQDN(x) for x in df['domain']]

    #function to calculate length of subdomain and generate new column
    def length_subdomain(domain):
        ext = tldextract.extract(domain)
        length_subdomain = len(ext.subdomain)

        return length_subdomain
    df['subdomain_length'] = [length_subdomain(x) for x in df['domain']]

    #function to calculate count of upper cases character and generate new column
    def upper(domain):
        upper = 0
        for i in domain:
            if (i.isupper()):
                upper = upper + 1
        return upper
    df['upper'] = [upper(x) for x in df['domain']]


    #function to calculate count of lower cases character and generate new column
    def lower(domain):
        lower = 0
        for i in domain:
            if (i.islower()):
                lower = lower + 1
        return lower
    df['lower'] = [lower(x) for x in df['domain']]



    #function to calculate count of numbers and generate new column
    def numeric(domain):
        numeric = 0
        for i in domain:
            if (i.isnumeric()):
                numeric = numeric + 1
        return numeric
    df['numeric'] = [numeric(x) for x in df['domain']]



    #function to calculate entropy and generate new column
    def calcEntropy(domain):
        p, lens = Counter(domain), np.float(len(domain))
        return -np.sum(count / lens * np.log2(count / lens) for count in p.values())

    df['entropy'] = [calcEntropy(x) for x in df['domain']]



    #function to calculate count of special character and generate new column
    def special(domain):
        special, alphabets, digits = 0, 0, 0
        for i in domain:
            if (i.isalpha()):
                alphabets += 1
            elif (i.isdigit()):
                digits += 1
            else:
                special = special + 1
        return special
    df['special'] = [special(x) for x in df['domain']]

    #function to calculate count number of labels and generate new column
    def labels(domain):
        counter = 1
        for i in domain:
            if (i == '.'):
                counter = counter + 1
        return counter
    df['labels'] = [labels(x) for x in df['domain']]


    #function to calculate maximum labels and generate new column
    def max_labels(domain):
        label_len = []
        Labels = domain.split('.')
        for i in Labels:
            label_len.append(len(i))
        max_len = max(label_len)
        max_index = label_len.index(max_len)
        return (label_len[max_index])

    df['labels_max'] = [max_labels(x) for x in df['domain']]



    #function to calculate average labels and generate new column
    def avg_label(domain):
        label_len = []
        labels = domain.split('.')
        for i in labels:
            label_len.append(len(i))
        sum_labels = sum(label_len)

        average_len_labels = sum_labels / len(label_len)

        return average_len_labels
    df['labels_average'] = [avg_label(x) for x in df['domain']]



    #function to calculate longest word and generate new column
    def longest_word(domain):
        label_len = []
        labels = domain.split('.')
        for i in labels:
            label_len.append(len(i))
        max_len = max(label_len)
        max_index = label_len.index(max_len)
        return (labels[max_index])

    df['longest_word'] = [longest_word(x) for x in df['domain']]



    #function to calculate second level domain and generate new colum
    def sld(domain):
        ext_domain = tldextract.extract(domain)
        sld = ext_domain.domain
        return sld

    df['sld'] = [sld(x) for x in df['domain']]


    #function to calculate length of domain and subdomain and generate new column
    def length_dom_sub(domain):
        ext = tldextract.extract(domain)
        length_domain = len(ext.domain)
        length_subdomain = len(ext.subdomain)
        lenn = length_domain + length_subdomain
        return lenn

    df['len'] = [length_dom_sub(x) for x in df['domain']]



    #function to check if there is subdomain or not and generate new column
    def sub_domain(domain):
        ext = tldextract.extract(domain)
        sub_domain = ext.subdomain
        len_sub_domain = len(sub_domain)
        if len_sub_domain > 0:
            return 1
        else:
            return 0

    df['subdomain'] = [sub_domain(x) for x in df['domain']]


    df.to_csv('../data/all_features.csv', index=False)


    #drop features and save dataframe in csv file
    df = df.drop(['domain', 'longest_word', 'sld'], axis=1)
    df.to_csv('../data/features.csv', index=False)



    # References:
    # https://pypi.org/project/tldextract/ tldextract
    # https://kldavenport.com/detecting-randomly-generated-domains/ entropy
    #
