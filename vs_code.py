import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with st.sidebar:
  st.title("Projet Energie")
  st.markdown("Groupe Décembre 2024")
st.write("Colonnes du DataFrame:", df_clean.columns.tolist())

@st.cache_data
def load_data_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?id=1b_yXHGs5Ms4O3wmUrxTLAXiJPHLm2ZXS"
    return pd.read_csv(url,sep=";")

file_id = "1b_yXHGs5Ms4O3wmUrxTLAXiJPHLm2ZXS"

try:
    df_clean = load_data_from_gdrive(file_id)
    st.success("✅ Données chargées depuis Google Drive.")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
  

df_clean['renouvelable'] = df_clean[['eolien', 'solaire', 'hydraulique', 'pompage', 'bioenergies']].sum(axis=1)
df_clean['non_renouvelable'] = df_clean[['thermique', 'nucleaire']].sum(axis=1)
df_clean['production_totale'] = df_clean['renouvelable'] + df_clean['non_renouvelable']


st.sidebar.title("Sommaire")
pages=["Présentation du jeu de données", "Exploration et Pre-processing", "Enrichissement de la base", "Machine Learning"]
page=st.sidebar.radio("Aller vers", pages)



if page == pages[0] : 
  st.title("Introduction")
  introduction = """
    L’objectif est de constater le **phasage entre la consommation et la production énergétique** au niveau national et régional (risque de *black-out* notamment).

    Pour ce faire, nous avons analysé la consommation d’énergies au niveau régional, par filière de production, en nous concentrant sur les **énergies renouvelables**, qui correspondent au **défi du 21ème siècle** en raison du réchauffement climatique.

    Le jeu de données est issu de la source de données de **l’ODRE** (Open Data Réseaux Énergies) avec un accès à toutes les informations de consommation et de production par filière, par jour (toutes les 1/2 heures) depuis **2013**.

    Les données proviennent de l’application **éCO2mix** et sont mises à disposition via [data.gouv.fr](https://data.gouv.fr) ou [odre.opendatasoft.com](https://odre.opendatasoft.com).

    Elles sont élaborées à partir des comptages et complétées par des forfaits. Les données sont dites **consolidées** lorsqu'elles ont été vérifiées et complétées (livraison en milieu de M+1).
    """
  st.markdown(introduction)
  st.dataframe(df_clean.head(10))
  

 

  
if page == pages[1]:
    st.title("Exploration des données")
    
    tab1, tab2 = st.tabs(["Exploration", "Visualisation"])
    
    with tab1:
        st.header("Exploration")
        st.dataframe(df_clean.describe())
        st.write("Voici les étapes d'exploration et de nettoyage des données :")
        st.write("1. Formatage des types de données")
        st.write("2. Contrôle de la cohérence des régions")
        st.write("3. Maintien des données définitives")
        st.write("4. Gestion des valeurs manquantes")
        st.write("5. Vérification des doublons")
    
with tab2:
    st.header("Visualisation")
    option = st.selectbox(
        "Choisissez une visualisation après traitement des données :",
        (
            "Consommation d'énergie par région",
            "Répartition de la production en France",
            "Production vs Consommation",
            "Focus production Nucléaire",
            "Focus énergie renouvelable"
        )
    )

    if option == "Consommation d'énergie par région":
        st.subheader("Consommation d'énergie par région")
        fig, ax = plt.subplots(figsize=(14, 6))
        df_clean.groupby('date')['consommation'].sum().plot(ax=ax)
        ax.set_title("Consommation d'énergie")
        ax.set_xlabel("Date")
        ax.set_ylabel("Consommation (MWh)")
        fig.tight_layout()
        st.pyplot(fig)

    elif option == "Répartition de la production en France":
        moyenne_production = df_clean[['nucleaire', 'thermique', 'hydraulique', 'eolien', 'solaire', 'bioenergies']].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(moyenne_production.index, moyenne_production.values, color='skyblue')
        ax.set_ylabel('Production moyenne (en MW)')
        ax.set_xlabel('Filières')
        ax.set_title('Répartition moyenne de la production par filière')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif option == "Production vs Consommation":
        st.subheader("Production vs Consommation")
        df_grouped = df_clean.groupby('annee')[['production_totale', 'consommation']].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_grouped, x='annee', y='production_totale', label='Production totale', color='green', marker='o', ax=ax)
        sns.lineplot(data=df_grouped, x='annee', y='consommation', label='Consommation', color='blue', marker='o', ax=ax)
        ax.set_title("Production vs Consommation par an")
        ax.set_xlabel("Année")
        ax.set_ylabel("Énergie (MWh)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    elif option == "Focus production Nucléaire":
        st.subheader("Focus production Nucléaire")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            x='annee', y='nucleaire', hue='region',
            data=df_clean, linewidth=2.5, palette="tab20", errorbar=None, ax=ax
        )
        ax.set_ylabel("")
        ax.set_title("Tendance nucléaire en MW par année et par région en France")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)

    elif option == "Focus énergie renouvelable":
        st.subheader("Focus énergie renouvelable")
        df_long = pd.melt(
            df_clean,
            id_vars=["annee"],
            value_vars=["hydraulique", "eolien", "solaire", "bioenergies"],
            var_name="source", value_name="production"
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='annee', y='production', hue='source', data=df_long, ax=ax)
        ax.set_title("Focus production renouvelable en MW en France par an par filière")
        ax.set_xlabel("Année")
        ax.set_ylabel("Production (MWh)")
        ax.legend(title="Source")
        fig.tight_layout()
        st.pyplot(fig)

  




