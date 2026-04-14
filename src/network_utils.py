import pandas as pd
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import random


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
            'timezone','timestamp', 'Venue_Category_Name'
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

# Fonction pour la construction des réseaux
def construction_reseau(dfjoint_complet, df_check_in):
    """
    dfagg_complet : contient la colonne 'checkins' (listes d'IDs) avec la liste de tous les check_ins par lieu
    df_check_in : contient 'location_id' et 'location_type_name'
    """
    df = dfjoint_complet.copy()

    # Création des arêtes
    df['edges'] = df['checkins'].apply(
        lambda x: list(itertools.combinations(x, 2))
    )
    edges = [edge for sublist in df['edges'] for edge in sublist]

    # Création du graphe
    G = nx.Graph()
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Ajout des attributs (type d'établissement)
    dict_categories = dict(zip(df_check_in['location_id'], 
                               df_check_in['category_grouped']))
    
    # Assignation
    nx.set_node_attributes(G, dict_categories, 'category_grouped')

    return G
    
    
# Calcul de plusieurs métriques sur le réseau : nbre de noeuds, d'arêtes, densité, degré moyen,
# distance moyenne (approximation)
def afficher_infos_reseau(G, titre, k=100):
    """
    G: réseau
    titre de l'affichage (ex: "caractéristiques du réseau macro")
    k : nombre de nœuds à échantillonner pour l'approximation de la distance moyenne.
    """
    print("\n" + "="*50)
    print(f"{titre}")
    print("="*50)
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    print(f"Nombre de lieux (nœuds) : {n_nodes}")
    print(f"Nombre de connexions (arêtes) : {n_edges}")
    print(f"Densité du réseau : {nx.density(G):.4f}")
    
    deg_moyen = sum(dict(G.degree()).values()) / n_nodes
    print(f"Degré moyen : {deg_moyen:.2f}")
    
    is_conn = nx.is_connected(G)
    print(f"Réseau entièrement connecté : {is_conn}")
    
    # Approximation de la distance moyenne au sein du réseau
    if is_conn:
        # Si le réseau est petit, on garde le calcul exact
        if n_nodes <= k:
            avg_path = nx.average_shortest_path_length(G)
            print(f"Distance moyenne (exacte) : {avg_path:.2f}")
        else:
            # On tire k nœuds au hasard
            nodes_sample = random.sample(list(G.nodes()), k)
            total_dist = 0
            for node in nodes_sample:
                # Calcule les distances du nœud source vers TOUS les autres
                paths = nx.single_source_shortest_path_length(G, node)
                total_dist += sum(paths.values())
            
            # Moyenne = (Somme des distances) / (nombre de chemins calculés)
            avg_path_approx = total_dist / (k * (n_nodes - 1))
            print(f"Distance moyenne (approx. avec k={k}) : {avg_path_approx:.2f}")
    else:
        print("Distance moyenne : non définie (réseau non connecté)")


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
def degree_distribution(G, titre='Distribution des degrés', color='#2A9D8F', markersize=8, bins=50):

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


# Fonction pour tracer un histogramme présentant la part de chaque catégorie d'établissement par intervalle de degrés
def plot_stacked_degree_categories(G, n_bins=15, top_n_categories=23):
    # Extraire les données
    data = []
    for node, attrs in G.nodes(data=True):
        deg = G.degree(node)
        cat = attrs.get('category_grouped', 'Unknown') 
        data.append({'degree': deg, 'category_grouped': cat})
    
    df = pd.DataFrame(data)

    #  Créer les bins logarithmiques
    kmin, kmax = df['degree'].min(), df['degree'].max()
    if kmin <= 0: kmin = 1 
    
    # On crée les limites des bacs (bins)
    bins = np.logspace(np.log10(kmin), np.log10(kmax), num=n_bins)
    
    # Assigner chaque degré à un bin (on utilise les intervalles comme labels)
    df['bin'] = pd.cut(df['degree'], bins=bins, include_lowest=True)
    
    #  Table croisée
    ct = pd.crosstab(df['bin'], df['category_grouped'], normalize='index')
    
    #  Tracé
    ax = ct.plot(kind='bar', stacked=True, figsize=(14, 7), width=0.8, colormap='Set3', edgecolor='black')

    plt.xlabel("Degré de connectivité (k)", size=12)
    
    # Graduation axe des abscisses : on garde l'intervalle complet mais on le formate (arrondi)
    new_labels = [f"[{int(b.left)}-{int(b.right)}]" for b in ct.index]
    
    ax.set_xticklabels(new_labels)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    for label in ax.get_xticklabels():
        label.set_visible(True)
    # ---------------------------------
    plt.legend(
    title="Groupes",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,   # ou 3 selon le nombre de catégories
    frameon=False
    )
    plt.tight_layout()
    plt.show()