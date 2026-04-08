import pandas as pd
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour la constitution des df dfjoint_complet (réseau complet)
# dfjoint_micro (réseau ne contenant que les liens créés par les utilisateurs ayant moins (<=) de 100 followers
# dfjoint_macro (réseau ne contenant que les liens créés par les utilisateurs ayant moins (>=) de 1000 followers

def constitution_df(path_data_check_in, path_data_users):
    
    # Lecture des données
    df_check_in = pd.read_csv(
        path_data_check_in,
        sep='\t',
        encoding='latin-1',
        header=None,
        names=[
            'user_id','location_id','location_type_ID',
            'location_type_name','latitude','longitude',
            'timezone','timestamp'
        ]
    )

    df_users = pd.read_csv(
        path_data_users,
        sep='\t',
        encoding='latin-1',
        header=None,
        names=[
            'user_id','gender','nb_twitter_friends',
            'nb_twitter_followers'
        ]
    )

    # Agrégation des check-ins
    dfagg_complet = (
        df_check_in
        .groupby('user_id', as_index=False)['location_id']
        .agg(list)
        .rename(columns={'location_id': 'checkins'})
    )

    # Nombre de check-ins
    dfagg_complet['n_checkins'] = dfagg_complet['checkins'].apply(len)

    # Filtre : au moins 2 check-ins (condition nécessaire pour créer un lien entre deux établissements)
    dfagg_complet = dfagg_complet[dfagg_complet['n_checkins'] > 1]

    # Jointure
    dfagg_complet['user_id'] = dfagg_complet['user_id'].astype(str)
    df_users['user_id'] = df_users['user_id'].astype(str)
    dfjoint_complet = dfagg_complet.merge(df_users, on='user_id', how='left')

    # Réseaux micro et macro
    dfjoint_micro = dfjoint_complet[dfjoint_complet['nb_twitter_followers'] <= 100]
    dfjoint_macro = dfjoint_complet[dfjoint_complet['nb_twitter_followers'] >= 2000]

    return dfjoint_complet, dfjoint_micro, dfjoint_macro


# Fonction pour la création du réseau (noeuds= établissements, arêtes= deux établissements sont liés 
# si ils ont été visités par un même utilisateur)

def construction_reseau(df):
    df = df.copy()

    # Étape 1 : toutes les paires de lieux par utilisateur
    df['edges'] = df['checkins'].apply(
        lambda x: list(itertools.combinations(x, 2))
    )

    # Étape 2 : flatten
    edges = [edge for sublist in df['edges'] for edge in sublist]

    # Étape 3 : création du graphe
    G = nx.Graph()
    G.add_edges_from(edges)

    # Étape 4 : suppression des self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


# Préparation des données pour un histogramme en échelle logarithmique
def logBinning(degreeList,nbin):
    kmin=min(degreeList)
    kmax=max(degreeList)
    logBins = np.logspace(np.log10(kmin), np.log10(kmax),num=nbin)
    logBinDensity, binedges = np.histogram(degreeList, bins=logBins, density=True)
    logBins = np.delete(logBins, -1)
    return logBinDensity, logBins


# Ajustement d'une loi puissance : f(x)= bx^a
def powerLaw(x, a, b):
    return b*x**(a)


# Tracé de la distribution des degrés des noeuds du réseau
def degree_distribution(G, titre='Distribution des degrés', color='#EB7009', markersize=8, bins=50):

    # Liste des degrés des nœuds
    kDict = dict(G.degree())
    kValues = list(kDict.values())
    
    # Binning logarithmique
    pk, k = logBinning(kValues, bins)
    
    # Tracé
    plt.figure(figsize=(8,6))
    plt.loglog(k, pk, 'o', color=color, markersize=markersize)
    plt.xlabel('Degré k', size=15)
    plt.ylabel('P(k)', size=15)
    plt.title(titre, size=16, weight='bold')
    plt.show()