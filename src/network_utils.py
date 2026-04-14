import pandas as pd
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

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
    dfjoint_macro = dfjoint_complet[dfjoint_complet['nb_twitter_followers'] >= 1000]

    return dfjoint_complet, dfjoint_micro, dfjoint_macro


# Fonction pour la création du réseau (noeuds= établissements, arêtes= deux établissements sont liés 
# si ils ont été visités par un même utilisateur)

def construction_reseau(df):
    df = df.copy()

    # Toutes les paires de lieux par utilisateur
    df['edges'] = df['checkins'].apply(
        lambda x: list(itertools.combinations(x, 2))
    )

    edges = [edge for sublist in df['edges'] for edge in sublist]

    # Création du graphe
    G = nx.Graph()
    G.add_edges_from(edges)

    # Suppression des self-loops
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
def degree_distribution(G, titre='Distribution des degrés', color='#1B263B', markersize=8, bins=50):

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

# Rregroupement les données dans des log-bins et nous traçons la moyenne de $knn$ pour chaque log-bin.
def plot_knn_logbins(G, titre='Degré moyen des voisins <knn>(k)', num_bins=15, alpha=0.1, marker_color='#1B263B', marker_size=10):
   
    # Calcul des degrés et knn
    kDict = dict(G.degree())
    knn = nx.average_neighbor_degree(G)
    
    # Extraction des valeurs pour tracé
    xx = [kDict[n] for n in knn.keys()]
    yy = [knn[n] for n in knn.keys()]
    
    # Tracé des points bruts
    plt.figure(figsize=(8,6))
    plt.loglog(xx, yy, 'o', alpha=alpha, color=marker_color)
    
    # Bins logarithmiques
    logBins = np.logspace(np.log2(np.min(xx)), np.log2(np.max(xx)), base=2, num=num_bins)
    
    # Moyenne des knn par bin
    ybin, xbin, binnumber = scipy.stats.binned_statistic(xx, yy, statistic='mean', bins=logBins)
    
    # Tracé des moyennes par bin
    plt.loglog(xbin[:-1], ybin, 'o', markersize=marker_size, color='black')
    
    # Labels et titre
    plt.xlabel('k', size=15)
    plt.ylabel('knn(k)', size=15)
    plt.title(titre, size=16, weight='bold')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()