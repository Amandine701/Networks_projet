import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def import_location_data(path):
    df = pd.read_csv(
        path,
        sep='\t',
        encoding='latin-1',
        header=None,
        names=[
            'user_id', 'location_id', 'location_type_ID',
            'location_type_name', 'latitude', 'longitude',
            'timezone', 'timestamp'
        ]
    )

    index_col = ["location_id", "latitude", "longitude", "location_type_ID", "location_type_name"]
    df = df.groupby(index_col)["user_id"].agg(pd.Series.nunique)
    df = df.reset_index().rename(columns={"user_id": "nb_checkins"})

    return df


def plot_type_repartion_log(df, filtered_types):
    # 1. Préparation des données et des bins
    df_clean = df[df["location_type_ID"].isin(filtered_types)].copy()
    df_clean['log_x'] = np.log10(df_clean['nb_checkins'].clip(lower=1))

    num_bins = 15
    bins = np.linspace(df_clean['log_x'].min(), df_clean['log_x'].max(), num_bins + 1)
    df_clean['bin_mid'] = pd.cut(df_clean['log_x'], bins=bins, include_lowest=True).apply(lambda x: x.mid)

    # 2. Calcul des volumes et des proportions
    # On compte par bin et par type
    df_plot = df_clean.groupby(['bin_mid', 'location_type_name']).size().reset_index(name='count_type')

    # On calcule le total par bin
    df_totals = df_plot.groupby('bin_mid')['count_type'].transform('sum')
    df_plot['total_bin'] = df_totals

    # On calcule la hauteur proportionnelle basée sur le log du total
    df_plot['display_height'] = (df_plot['count_type'] / df_plot['total_bin']) * np.log10(df_plot['total_bin'].replace(0, 1))

    # 3. Création du graphique (Axe Y linéaire dans le code, mais visuellement Log)
    fig = px.bar(
        df_plot,
        x='bin_mid',
        y='display_height',
        color='location_type_name',
        title="Repartition of check-ins per location (log-log)",
        subtitle="Internal repartition of location type in linear scale. Only the most visited types are represented.",
        # On ajoute les vraies valeurs dans le survol (hover) pour que ce soit lisible
        hover_data={'bin_mid': False, 'display_height': False, 'count_type': True, 'total_bin': True},
        labels={'count_type': 'Count', 'total_bin': 'Total Bin', 'bin_mid': 'Check-ins', 'location_type_name': 'Type'}
    )

    # 4. Configuration des axes pour simuler le Log-Log
    # Axe X
    x_min, x_max = int(df_clean['log_x'].min()), int(df_clean['log_x'].max())
    x_ticks = np.arange(x_min, x_max + 1)

    # Axe Y (On simule l'échelle log car display_height est déjà un log)
    y_max = int(np.ceil(df_plot['display_height'].groupby(df_plot['bin_mid']).sum().max()))
    y_ticks = np.arange(0, y_max + 1)
    y_text = [f"{10**v}" for v in y_ticks]

    fig.update_layout(
        bargap=0,
        xaxis=dict(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=[str(10**v) for v in x_ticks],
            title="Number of check-ins (Log)"
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=y_ticks,
            ticktext=y_text,
            title="Number of locations (Log)"
        )
    )

    fig.update_traces(marker_line_width=0)

    return fig


def preprocess_data(df):
    X = df[["latitude", "longitude", "nb_checkins"]].values
    X = StandardScaler().fit_transform(X)
    return X
